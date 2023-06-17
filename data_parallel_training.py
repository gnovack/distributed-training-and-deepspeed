import argparse
import os
import time
import torch.multiprocessing as mp
from tqdm import tqdm
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import BertForMaskedLM, AdamW, AutoTokenizer, DataCollatorForLanguageModeling

from util import load_wikitext, get_device_count


def create_process_group(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank
    )


def train(rank, world_size, batch_size, training_steps, bucket_size, model):

    create_process_group(rank, world_size)
    
    model_name = "bert-large-cased" if model == "large" else "bert-base-cased"
    model = BertForMaskedLM.from_pretrained(model_name).to(rank)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    collator = DataCollatorForLanguageModeling(tokenizer)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    train_dataset = load_wikitext(tokenizer, collator).select(range(batch_size * training_steps))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=DistributedSampler(train_dataset),
    )
    
    model = DistributedDataParallel(model, device_ids=[rank], bucket_cap_mb=bucket_size)
    model.train()

    training_start_time = time.time()
    progress_bar = tqdm(range(training_steps))

    for batch in train_dataloader:
        input_ids = batch['input_ids']

        outputs = model(input_ids, labels=batch['labels'])
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        progress_bar.update(1)
    
    training_end_time = time.time()
    print(f"\nTotal Training Time: {training_end_time - training_start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--training-steps", type=int, default=1000)
    parser.add_argument("--device-count", type=int, default=None)
    parser.add_argument("--bucket-size", type=int, default=25)
    parser.add_argument("--model", type=str, choices=["base", "large"], default="base")
    args = parser.parse_args()

    device_count = args.device_count or get_device_count()
    mp.spawn(
        train, 
        args=(device_count, args.batch_size, args.training_steps, args.bucket_size, args.model), 
        nprocs=device_count, 
        join=True
    )