import argparse
import deepspeed
import deepspeed.comm as dist
import os
import time
import torch
import transformers  
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling

from util import load_wikitext


def train(batch_size, training_steps, stage, *args):
    transformers.logging.set_verbosity_warning()
    rank = int(os.getenv("LOCAL_RANK", "0"))
    device = torch.device("cuda", rank)

    model = BertForMaskedLM.from_pretrained("bert-large-cased")
    tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")
    collator = DataCollatorForLanguageModeling(tokenizer)

    train_dataset = load_wikitext(tokenizer, collator).select(range(batch_size* training_steps))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

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
            "contiguous_gradients": True
        }
    }

    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )

    model_engine.train()
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
            progress_bar.update(1)
        
    
    dist.log_summary()
    training_end_time = time.time()
    if rank == 0:
        print(f"\nTotal Training Time: {training_end_time - training_start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--training-steps", type=int, default=100)
    parser.add_argument("--stage", type=int, default=1)
    args, extra_args = parser.parse_known_args()
    
    train(args.batch_size, args.training_steps, args.stage, *extra_args)