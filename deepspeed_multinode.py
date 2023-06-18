import argparse
import deepspeed
import deepspeed.comm as dist
import os
import time
import torch
import transformers
from deepspeed.runtime.utils import memory_status
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import OPTForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

from util import load_wikitext


def train(batch_size, training_steps, stage, *args):
    transformers.logging.set_verbosity_warning()
    rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    
    device = torch.device("cuda", rank)

    model = OPTForCausalLM.from_pretrained("facebook/opt-125m")

    ds_config = {
        "train_micro_batch_size_per_gpu": batch_size,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.00015
            }
        },
        "comms_logger": {
            "enabled": True,
            "verbose": False,
            "prof_all": True,
            "debug": False,
        },
        "zero_optimization": {
            "stage": stage,
            "reduce_bucket_size": 5e6,
        }
    }

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )

    print(f"Device {rank} - ZeRO Stage: {model_engine.zero_optimization_stage()}")

    optimizer_state = optimizer.param_groups[0]
    print(f"Device {rank} - Optimizer: lr={optimizer_state['lr']}; "
          f"betas={optimizer_state['betas']}; eps={optimizer_state['eps']}; "
          f"parameter count={sum([torch.numel(p) for p in optimizer_state['params']]):,}")
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_dataset = load_wikitext(tokenizer, collator, max_length=512).select(range(batch_size * training_steps))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=DistributedSampler(train_dataset, num_replicas=world_size)
    )

    model_engine.train()
    
    # reset communcations logs so that only communications
    # made during the training loop will be reflected in 
    # total_comms_latency below
    dist.comms_logger.comms_dict = {}

    training_start_time = time.time()
    
    if rank == 0:
        progress_bar = tqdm(range(training_steps))

    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model_engine(input_ids, labels=labels)
        loss = outputs.loss
        
        model_engine.backward(loss)

        model_engine.step()

        if rank == 0:
            memory_status("Memory stats after training step:")
            progress_bar.update(1)
        
    
    training_end_time = time.time()
    if rank == 0:
        print(f"\nTotal Training Time: {training_end_time - training_start_time:.2f} seconds")

    # print average comms latency across all comms
    total_comms_latency = 0
    for op_name in dist.comms_logger.comms_dict.keys():
        if op_name == "log_summary_barrier":
            continue
        for _, vals in sorted(dist.comms_logger.comms_dict[op_name].items()):
            total_comms_latency += sum(vals[1])
    
    dist.log_summary()

    if rank == 0:
        print(f"\nTotal Communication Latency: {total_comms_latency:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--training-steps", type=int, default=100)
    parser.add_argument("--stage", type=int, default=0)
    args, extra_args = parser.parse_known_args()

    print("Extra args: ", extra_args)
    
    train(args.batch_size, args.training_steps, args.stage, *extra_args)