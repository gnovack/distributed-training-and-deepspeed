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
