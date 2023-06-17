import torch


def get_model_memory(model: torch.nn.Module):
    """
    Returns the memory usage of the given model
    """
    total_memory = 0
    for param in model.parameters():
        total_memory += param.numel() * param.element_size()
    return total_memory


class ActivationCounter:
    """Helper class to count the number of activations in a model."""

    def __init__(self):
        self.activation_bytes = 0

    def add_activations(self, tensor):
        self.activation_bytes += tensor.numel() * tensor.element_size()


def activation_counter_hook(counter: ActivationCounter):
    """Returns a hook that counts the number of activations."""

    def hook(self, _, output):
        counter.add_activations(output.data)

    return hook


def register_hooks_recursive(model: torch.nn.Module, counter: ActivationCounter):
  """Recursively injects activation counting hooks into the given model."""
  for module in model.children():
      module.register_forward_hook(activation_counter_hook(counter))
      register_hooks_recursive(module, counter)


if __name__ == "__main__":
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 1024),
        torch.nn.Linear(1024, 1024),
        torch.nn.Linear(1024, 1024),
        torch.nn.Linear(1024, 512)
    )

    print("Model Memory: {:,} bytes".format(get_model_memory(model)))

    activation_counter = ActivationCounter()
    register_hooks_recursive(model, activation_counter)

    inputs = torch.randn(4, 512)
    outputs = model(inputs)

    # because the hooks only capture layer outputs, we need to add
    # the size of the original input tensor separately
    activation_counter.add_activations(inputs)

    print("Activation Memory: {:,} bytes".format(
        activation_counter.activation_bytes
    ))