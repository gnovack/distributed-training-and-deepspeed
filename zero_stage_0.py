import argparse
import deepspeed
import deepspeed.comm as dist
import os
import time
import torch
import transformers  
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import BertForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, BloomForCausalLM

from util import load_wikitext


def main():
    
    rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m")
        
    deepspeed_config = {
        "train_micro_batch_size_per_gpu": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 5e-5
            }
        },
        "fp16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "allgather_bucket_size": 5e4,
            "reduce_bucket_size": 5e5,
            "stage3_max_reuse_distance": 1e4,
            "stage3_max_live_parameters": 1e4,
            "stage3_prefetch_bucket_size": 5e4,
            "stage3_param_persistence_threshold": 1e4,
            "sub_group_size": 1e4
        }
    }

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=deepspeed_config
    )

    print(f"Device {rank} - ZeRO Stage: {model_engine.zero_optimization_stage()}")

    print(f"Device {rank} - Model parameters: {sum([torch.numel(p) for p in model_engine.parameters()]):,}")
    # print(f"Device {rank} - Params: {sum([torch.numel(x) for x in model_engine.optimizer.fp32_partitioned_groups_flat])}")

    optimizer_state = optimizer.param_groups[0]
    print(f"Device {rank} - Optimizer: lr={optimizer_state['lr']}; "
          f"betas={optimizer_state['betas']}; eps={optimizer_state['eps']}; "
          f"parameter count={sum([torch.numel(p) for p in optimizer_state['params']]):,}")

    model_engine.train()

    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    train_dataset = load_wikitext(tokenizer, collator, max_length=512).select(range(128))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        sampler=DistributedSampler(train_dataset, num_replicas=world_size)
    )

    device = torch.device("cuda", rank)

    from deepspeed.runtime.utils import memory_status
    
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model_engine(input_ids, labels=labels)
        
        model_engine.backward(outputs.loss)

        # stage = 2
        # if stage == 2 and rank in model_engine.optimizer.averaged_gradients:
        #     print(f"Rank {rank} - Gradients: {sum([torch.numel(p) for p in model_engine.optimizer.averaged_gradients[rank]])}")
        
        model_engine.step()

        memory_status("Memory stats after training step")
        

        

    return


if __name__ == "__main__":
    main()