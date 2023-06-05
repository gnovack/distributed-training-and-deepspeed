import argparse
import transformers
import torch
import time
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForMaskedLM, AdamW, AutoTokenizer, DataCollatorForLanguageModeling

from util import load_wikitext, get_device, get_device_count

if __name__ == "__main__":
    transformers.logging.set_verbosity_warning()

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--training-steps", type=int, default=100)
    parser.add_argument("--device-count", type=int, default=None)
    args = parser.parse_args()

    model = BertForMaskedLM.from_pretrained("bert-base-cased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    collator = DataCollatorForLanguageModeling(tokenizer)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_dataset = load_wikitext(tokenizer, collator)
    train_dataloader = DataLoader(
        train_dataset.select(range(args.batch_size*args.training_steps)),
        shuffle=True,
        batch_size=args.batch_size
    )

    device_type = get_device()
    device_count = args.device_count or get_device_count()

    print("Using device: {}".format(device_type))
    if device_count > get_device_count():
        print(f"Cannot use {device_count} {device_type} devices. Only {get_device_count()} devices are available. "
                "Training will be done using {get_device_count()} devices.")
        device_count = get_device_count()
    
    model = DistributedDataParallel(model, device_ids=list(range(device_count)))
    
    model.train()

    training_start_time = time.time()
    if not args.verbose:
        progress_bar = tqdm(range(args.training_steps))

    for batch in train_dataloader:
        input_ids = batch['input_ids']

        outputs = model(input_ids, labels=batch['labels'])

        # labels = batch['labels']
        # loss = loss_fn(outputs.view(-1, config.vocab_size), labels.view(-1))
        loss = outputs.loss

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if not args.verbose:
            progress_bar.update(1)
        
    training_end_time = time.time()

    print(f"\nTotal Training Time: {training_end_time - training_start_time:.2f} seconds")

