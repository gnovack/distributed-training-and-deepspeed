import argparse
import time
from typing import Optional
import datasets
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, BertConfig

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--training-steps", type=int, default=100)
    parser.add_argument("--device-count", type=Optional[int], default=None)
    parser.add_argument("--micro-batch-count", type=int, default=8)
    args = parser.parse_args()

    config = BertConfig.from_pretrained("bert-base-cased")
    model = BertModelWithMP(
        config=config,
        device_count=args.device_count,
        verbose=args.verbose
    )

    # store the input and output devices for later use.
    # this is required because embedding_device and head_device
    # will not be available after the model is converted to pipeline.
    input_device = model.embedding_device
    output_device = model.head_device

    if args.pipeline:
        model = model.to_pipeline()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    collator = DataCollatorForLanguageModeling(tokenizer)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_fn = torch.nn.CrossEntropyLoss().to(output_device)

    train_dataset = load_wikitext(tokenizer, collator)
    train_dataloader = DataLoader(
        train_dataset.select(range(args.batch_size*args.training_steps)),
        shuffle=True,
        batch_size=args.batch_size
    )
    model.train()

    training_start_time = time.time()
    progress_bar = tqdm(range(args.training_steps))
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(input_device)

        outputs = model(input_ids)
        if args.pipeline:
            outputs = outputs.local_value()

        labels = batch['labels'].to(output_device)
        loss = loss_fn(outputs.view(-1, config.vocab_size), labels.view(-1))

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if not args.verbose:
            progress_bar.update(1)
        
    training_end_time = time.time()
    print(f"Training time: {training_end_time - training_start_time} seconds")

