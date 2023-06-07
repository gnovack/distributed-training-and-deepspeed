import argparse
import deepspeed
import os
import time
import torch
import torch.multiprocessing as mp
import transformers  
from tqdm import tqdm
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import BertForMaskedLM, AdamW, AutoTokenizer, DataCollatorForLanguageModeling

from util import load_wikitext, get_device_count


def create_process_group(rank, world_size):

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    deepspeed.init_distributed(
        dist_backend='nccl',
        world_size=world_size,
        rank=rank
    )

def train(batch_size, training_steps):
    transformers.logging.set_verbosity_warning()
    rank = int(os.getenv("LOCAL_RANK", "0"))
    device = torch.device("cuda", rank)

    model = BertForMaskedLM.from_pretrained("bert-base-cased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    collator = DataCollatorForLanguageModeling(tokenizer)

    train_dataset = load_wikitext(tokenizer, collator).select(range(batch_size* training_steps))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    ds_config = {
        "train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
            "lr": 0.00015
            }
        },
        "fp16": {
            "enabled": False
        },
        "zero_optimization": True
    }
    
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )
    # model = DistributedDataParallel(model, device_ids=[rank], bucket_cap_mb=bucket_size)

    training_start_time = time.time()
    progress_bar = tqdm(range(training_steps))

    # profile = True
    # if profile:
    #     with torch.profiler.profile() as p:
    #         batch = next(iter(train_dataloader))
    #         input_ids = batch['input_ids']

    #         outputs = model_engine(input_ids, labels=batch['labels'])
    #         loss = outputs.loss
    #         loss.backward()

    #         optimizer.step()
    #         optimizer.zero_grad()
    #     p.export_chrome_trace("data-parallel-trace.json")
    #     return

    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model_engine(input_ids, labels=labels)
        loss = outputs.loss
        model_engine.backward(loss)

        model_engine.step()
        # loss.backward()

        # optimizer.step()
        # optimizer.zero_grad()

        progress_bar.update(1)
        
    training_end_time = time.time()

    print(f"\nTotal Training Time: {training_end_time - training_start_time:.2f} seconds")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--batch-size", type=int, default=8)
    # parser.add_argument("--training-steps", type=int, default=100)
    # parser.add_argument("--device-count", type=int, default=None)
    # parser.add_argument("--bucket-size", type=int, default=25)
    # args = parser.parse_args()

    # device_count = args.device_count or get_device_count()

    # mp.spawn(
    #     train, 
    #     args=(3, 32, 100), 
    #     nprocs=get_device_count(), 
    #     join=True
    # )
    train(8, 100)