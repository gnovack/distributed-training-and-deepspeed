import datasets
import math
import torch


def get_device():
    """Get the device to use for training."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_device_count():
    """Get the total number of available devices."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    else:
        return 1


def format_size(size_bytes):
   """
   Converts the given size in bytes to a human readable format
   Reference: https://stackoverflow.com/questions/5194057/better-way-to-convert-file-sizes-in-python
   """
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])


def load_wikitext(tokenizer, collator, max_length=None):

    def mask_tokens(x):
        input_ids, labels = collator.torch_mask_tokens(x['input_ids'], special_tokens_mask=x['special_tokens_mask'])
        return {
            "input_ids": input_ids,
            "labels": labels
        }

    wikitext = datasets.load_dataset("wikitext", "wikitext-2-v1")
    train_dataset = wikitext["train"]
    
    train_dataset = train_dataset.map(lambda x: tokenizer(x["text"], max_length=max_length, padding='max_length', truncation=True, return_tensors='pt', return_special_tokens_mask=True), batched=True)
    train_dataset.set_format(type="torch", columns=["input_ids", "special_tokens_mask"])
    if collator.mlm:
        train_dataset = train_dataset.map(mask_tokens, remove_columns=['special_tokens_mask'])
    else:
        train_dataset = train_dataset.map(lambda x: {
            "input_ids": x["input_ids"],
            "labels": x["input_ids"]
        })

    return train_dataset