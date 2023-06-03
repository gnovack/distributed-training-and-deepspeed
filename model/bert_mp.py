from datetime import datetime
import itertools
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
        self.last_timestamp = {}

        self.embeddings = BertEmbeddings(config)
        self.encoders = torch.nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.head = BertOnlyMLMHead(config)

        # total number of modules to parallelize (embeddings + encoders + head)
        num_modules = len(self.encoders) + 2

        # naive device placement logic which evenly distributes modules 
        # across all available devices
        device = get_device()
        device_count = device_count or get_device_count()
        
        print("Using device: {}".format(device))
        if device_count <= 1:
            print(f"Only one {device} device is available. Training will be done without Model Parallelism.")
        elif device_count > get_device_count:
            print(f"Cannot use {device_count} {device} devices. Only {get_device_count()} devices are available. "
                   "Training will be done using {get_device_count()} devices.")
            device_count = get_device_count()
            

        device_cycle = itertools.cycle(range(device_count))
        device_indices = sorted([next(device_cycle) for _ in range(num_modules)])
        devices = [torch.device(device, i) for i in device_indices]

        i = 0
        self.embedding_device = devices[i]
        self.embeddings.to(self.embedding_device)

        if self.verbose:
            self.embeddings.register_forward_hook(self.idle_time_hook(self.embedding_device))
        i += 1

        self.encoder_devices = []
        for encoder in self.encoders:
            device = devices[i]
            if device != devices[i-1] and self.verbose:
                encoder.register_forward_hook(self.idle_time_hook(device))
                encoder.register_full_backward_hook(self.idle_time_hook(devices[i-1], forward=False))
            self.encoder_devices.append(device)
            encoder.to(device)
            i += 1
        
        self.head_device = devices[i]
        self.head.to(self.head_device)

        if self.head_device != self.encoder_devices[-1] and self.verbose:
            self.head.register_forward_hook(self.forward_hook(self.head_device))
            self.head.register_full_backward_hook(self.idle_time_hook(self.encoder_devices[-1], forward=False))

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
        if self.verbose:
            outputs.register_hook(self.idle_time_hook(self.head_device, forward=False))
        
        return outputs


    def log(self, message):
        message = f"{datetime.now()} - {message}"
        print(message)

    def idle_time_hook(self, device, forward=True):
        """Creates a PyTorch hook which logs the idle time of a device."""
        def hook(*args, **kwargs):
            current_timestamp = time.time()
            last_timestamp = self.last_timestamp.get(device, None)

            if forward:
                message = f"Running forward pass on device {device}"
            else:
                message = f"Running backward pass on device {device}"

            if last_timestamp is not None:
                message += f". Idle time: {(current_timestamp - last_timestamp)*1000:.2f}ms"

            self.last_timestamp[device] = current_timestamp
            self.log(message)
        return hook
