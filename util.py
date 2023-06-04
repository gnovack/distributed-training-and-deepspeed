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