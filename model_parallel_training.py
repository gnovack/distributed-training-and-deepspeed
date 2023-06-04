import argparse
import time
import datasets
import torch
import transformers
from tabulate import tabulate
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, BertConfig
from typing import Optional

from model.bert_mp import BertModelWithMP


def load_wikitext(tokenizer, collator):

    def mask_tokens(x):
        input_ids, labels = collator.torch_mask_tokens(x['input_ids'], special_tokens_mask=x['special_tokens_mask'])
        return {
            "input_ids": input_ids,
            "labels": labels
        }

    wikitext = datasets.load_dataset("wikitext", "wikitext-2-v1")
    train_dataset = wikitext["train"]
    
    train_dataset = train_dataset.map(lambda x: tokenizer(x["text"], padding='max_length', truncation=True, return_tensors='pt', return_special_tokens_mask=True), batched=True)
    train_dataset.set_format(type="torch", columns=["input_ids", "special_tokens_mask"])
    train_dataset = train_dataset.map(mask_tokens, remove_columns=['special_tokens_mask'])

    return train_dataset


def summarize_idle_time(bert: BertModelWithMP, training_steps: int):
    avg_idle_times = [
        ["Device", "Average Idle Time (ms)"],
    ]
    for k, v in bert.device_idle_time.items():
        avg_idle_times.append([k, v[0] / training_steps])
        
    print(tabulate(avg_idle_times, headers="firstrow", floatfmt=".2f", tablefmt="fancy_grid"))

if __name__ == "__main__":
    transformers.logging.set_verbosity_warning()

    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--training-steps", type=int, default=100)
    parser.add_argument("--device-count", type=Optional[int], default=None)
    parser.add_argument("--micro-batch-count", type=int, default=8)
    args = parser.parse_args()

    config = BertConfig.from_pretrained("bert-base-cased")
    bert = BertModelWithMP(
        config=config,
        device_count=args.device_count,
        verbose=args.verbose
    )

    if args.pipeline:
        model = bert.to_pipeline(chunks=args.micro_batch_count)
    else:
        model = bert

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    collator = DataCollatorForLanguageModeling(tokenizer)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_fn = torch.nn.CrossEntropyLoss().to(bert.head_device)

    train_dataset = load_wikitext(tokenizer, collator)
    train_dataloader = DataLoader(
        train_dataset.select(range(args.batch_size*args.training_steps)),
        shuffle=True,
        batch_size=args.batch_size
    )
    model.train()

    training_start_time = time.time()
    if not args.verbose:
        progress_bar = tqdm(range(args.training_steps))

    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(bert.embedding_device)

        outputs = model(input_ids)
        if args.pipeline:
            outputs = outputs.local_value()

        labels = batch['labels'].to(bert.head_device)
        loss = loss_fn(outputs.view(-1, config.vocab_size), labels.view(-1))

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if not args.verbose:
            progress_bar.update(1)
        
    training_end_time = time.time()

    print(f"\nTotal Training Time: {training_end_time - training_start_time:.2f} seconds")
    
    print("\nAverage Idle Time per Device:")
    summarize_idle_time(bert, args.training_steps)
    
    

