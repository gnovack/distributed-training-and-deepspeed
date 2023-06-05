from datetime import datetime
import numpy as np
import tempfile
import time
import torch
from torch.distributed import rpc
from torch.distributed.pipeline.sync import Pipe
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings, BertLayer, BertOnlyMLMHead

from util import get_device, get_device_count


class BertModelWithMP(torch.nn.Module):
    """Minimal implementation of BERT with support for model and pipeline parallelism."""
    def __init__(self, config: BertConfig, device_count=None, verbose=False):
        super().__init__()
        self.config = config
        self.verbose = verbose
        
        self.devices = []
        self.embeddings = BertEmbeddings(config)
        self.encoders = torch.nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.head = BertOnlyMLMHead(config)

        # naive device placement logic which evenly distributes modules 
        # across all available devices
        device_type = get_device()
        device_count = device_count or get_device_count()

        print("Using device: {}".format(device_type))
        if device_count <= 1:
            print(f"Only one {device_type} device is available. Training will be done without Model Parallelism.")
        elif device_count > get_device_count():
            print(f"Cannot use {device_count} {device_type} devices. Only {get_device_count()} devices are available. "
                   "Training will be done using {get_device_count()} devices.")
            device_count = get_device_count()

        modules = [self.embeddings] + [e for e in self.encoders] + [self.head]
        modules_grouped = np.array_split(modules, device_count)

        for group, device_idx in zip(modules_grouped, range(device_count)):
            device = torch.device(device_type, device_idx)

            for module in group:
                self.devices.append(device)
                module.to(device)
            
            # add forward and backward hooks to track idle time
            group[0].register_forward_hook(self.idle_time_hook(device.index))
            group[-1].register_forward_hook(self.idle_time_hook(device.index, entering=False))

            group[0].register_full_backward_hook(self.idle_time_hook(device.index, forward=False, entering=False))
            group[-1].register_full_backward_hook(self.idle_time_hook(device.index, forward=False))

        # map used to track the last timestamp at which a device was
        # used during a forward or backward pass
        self.previous_timestamp = {}
        self.device_idle_time = {d: (0,0) for d in range(device_count)}

    @property
    def embedding_device(self):
        return self.devices[0]
    
    @property
    def encoder_devices(self):
        return self.devices[1:-1]
    
    @property
    def head_device(self):
        return self.devices[-1]
    
    def to_pipeline(self, chunks):
        """Convert the model for pipeline parallelism."""
        rpc.init_rpc(
            name="worker",
            rank=0,
            world_size=1,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                init_method="file://{}".format(tempfile.NamedTemporaryFile().name)
            )
        )

        sequential = torch.nn.Sequential(
            self.embeddings,
            *self.encoders, 
            self.head
        )
        return Pipe(sequential, chunks=chunks)

    def forward(self, input_ids: torch.LongTensor):

        hidden_states = self.embeddings(input_ids.to(self.embedding_device))
        for encoder, device in zip(self.encoders, self.encoder_devices):
            hidden_states = encoder(hidden_states.to(device))[0]

        outputs = self.head(hidden_states.to(self.head_device))
        
        return outputs

    def log(self, message):
        if self.verbose:
            message = f"{datetime.now()} - {message}"
            print(message)

    def idle_time_hook(self, device, forward=True, entering=True):
        """Creates a PyTorch hook which logs the idle time of a device."""
        
        def hook(*args, **kwargs):
            current_timestamp = time.time()
            last_timestamp = self.previous_timestamp.get(device, None)

            message = "{} {} pass on device {}".format(
                "Entering" if entering else "Finished",
                "forward" if forward else "backward",
                device
            )

            if entering and last_timestamp is not None:
                idle_time_ms = (current_timestamp - last_timestamp) * 1000
                self.device_idle_time[device] = (
                    self.device_idle_time[device][0] + idle_time_ms,
                    self.device_idle_time[device][1] + 1
                )
                
                message += f". Idle time: {idle_time_ms:.2f}ms"

            self.previous_timestamp[device] = current_timestamp
            self.log(message)
        return hook
