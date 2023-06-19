import argparse
import time
import torch
import transformers
from tabulate import tabulate
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, BertConfig

from model.bert_mp import BertModelWithMP
from util import load_wikitext


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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--training-steps", type=int, default=250)
    parser.add_argument("--device-count", type=int, default=None)
    parser.add_argument("--micro-batch-count", type=int, default=4)
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
