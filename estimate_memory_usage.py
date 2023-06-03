import math
import torch
from model.transformer import TransformerBlock
from transformers.models.opt.modeling_opt import OPTConfig


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        raise ValueError(f"Unsupported device")


def get_current_memory_allocation(device):
    if device == "mps":
        return torch.mps.current_allocated_memory()
    elif device == "cuda":
        return torch.cuda.memory_allocated()
    else:
        raise ValueError(f"Unsupported device: {device}")


def convert_size(size_bytes):
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


def get_model_memory(model: torch.nn.Module):
    """
    Returns the memory consumed by the parameters of the given model.
    """
    total_memory = 0
    for param in model.parameters():
        total_memory += param.numel() * param.element_size()
    return total_memory

def get_optimizer_memory(model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Returns the memory usage (in bytes) of the given optimizer and model.
    Note: Currently only supports SGD, Adam, and AdamW.
    """
    model_parameters = sum(param.numel() for param in model.parameters())
    
    # at minimum, 4 bytes are required to store
    # the gradient for each parameter
    bytes_per_param = 4

    if type(optimizer) == torch.optim.SGD:
        has_momentum = any(param_group.get('momentum', 0) != 0 
                               for param_group in optimizer.param_groups)
        if has_momentum:
            bytes_per_param += 4
    elif type(optimizer) in (torch.optim.Adam, torch.optim.AdamW):
        bytes_per_param += 8
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    
    return model_parameters * bytes_per_param


def load_transformer(config):
    """
    Loads a transformer block with the given configuration.
    """
    return TransformerBlock(config)


class ActivationCounter:
    """
    Helper class used to count the number of activations during
    a forward pass.
    """

    def __init__(self):
        self.activation_bytes = 0

    def add_activations(self, tensor):
        self.activation_bytes += tensor.numel() * tensor.element_size()


def activation_counter_hook(counter: ActivationCounter):
    """
    PyTorch forward hook that counts the number of activations 
    in a model forward pass.
    """

    def hook(self, input, output):
        counter.add_activations(output.data)

    return hook


def register_hooks_recursive(model, counter: ActivationCounter):
    for module in model.children():
        module.register_forward_hook(activation_counter_hook(counter))
        register_hooks_recursive(module, counter)


def get_optimizer_bytes_per_parameter(optimizer: torch.optim.Optimizer):
    """
    Returns the memory usage (in bytes) of the given optimizer and model.
    Note: Currently only supports SGD, Adam, and AdamW.
    """
    
    # at minimum, 4 bytes are required to store
    # the gradient for each parameter
    bytes_per_param = 4

    if type(optimizer) == torch.optim.SGD:
        has_momentum = any(param_group.get('momentum', 0) != 0 
                               for param_group in optimizer.param_groups)
        if has_momentum:
            bytes_per_param += 4
    elif type(optimizer) in (torch.optim.Adam, torch.optim.AdamW):
        bytes_per_param += 8
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    
    return bytes_per_param


def transformer_model_memory(
        layers, hidden_size, ffn_dim, num_attention_heads, 
        batch_size, sequence_length, optimizer):

    model_memory = 4 * layers * (
        9 * hidden_size + 
        4 * hidden_size ** 2 + 
        2 * hidden_size * ffn_dim + 
        ffn_dim
    )

    activation_memory = 4 * layers * batch_size * sequence_length * (
        10 * hidden_size +
        3 * num_attention_heads * sequence_length +
        2 * ffn_dim
    )

    optimizer_memory = (model_memory / 4) * get_optimizer_bytes_per_parameter(optimizer)

    return model_memory + activation_memory + optimizer_memory



if __name__ == "__main__":

    batch_size = 32

    config = OPTConfig(**{
        "hidden_size": 768,
        "num_attention_heads": 12,
        "enable_bias": True,
        "max_position_embeddings": 1024,
        "ffn_dim": 3072,
    })
    model = load_transformer(config)

    est_size = convert_size(transformer_model_memory(
        1, config.hidden_size, config.ffn_dim, 
        config.num_attention_heads, batch_size, 
        config.max_position_embeddings, torch.optim.Adam(model.parameters()))
    )
    print(f"Projected total memory usage (with formula): {est_size}")

    inputs = torch.randn(batch_size, config.max_position_embeddings, config.hidden_size)

    device = get_device()
    model.to(device)

    memory_allocation_with_model = get_current_memory_allocation(device)
    estimated_model_memory = get_model_memory(model)
    
    print(f"Consumed Model Memory: {convert_size(memory_allocation_with_model)}")
    print(f"Estimated Model Memory: {convert_size(estimated_model_memory)}")
    print(f"Percent difference: {abs(memory_allocation_with_model - estimated_model_memory) / estimated_model_memory * 100:.2f}%")
    print("-" * 80)

    activation_counter = ActivationCounter()
    register_hooks_recursive(model, activation_counter)

    inputs = inputs.to(device)
    outputs = model(inputs)

    activation_counter.add_activations(inputs)

    memory_allocation_forward_pass = get_current_memory_allocation(device) - memory_allocation_with_model
    estimated_activation_memory = 0
    print(f"Consumed Activation Memory: {convert_size(memory_allocation_forward_pass)}")
    print(f"Estimated Activation Memory: {convert_size(activation_counter.activation_bytes)}")
    print(f"Percent difference: {abs(memory_allocation_forward_pass - activation_counter.activation_bytes) / activation_counter.activation_bytes * 100:.2f}%")
    print("-" * 80)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    labels = torch.randn_like(outputs).to(device)
    labels_size = labels.numel() * labels.element_size()
    outputs_size = outputs.numel() * outputs.element_size()

    loss = loss_fn(outputs, labels)

    loss.backward(retain_graph=True)
    optimizer.step()

    post_backward_memory = get_current_memory_allocation(device) - memory_allocation_with_model - memory_allocation_forward_pass
    post_backward_memory = post_backward_memory - labels_size - outputs_size

    estimated_optimizer_memory = get_optimizer_memory(model, optimizer)

    print(f"Consumed Optimizer Memory: {convert_size(post_backward_memory)}" )
    print(f"Estimated Optimizer Memory: {convert_size(estimated_optimizer_memory)}")
    print(f"Percent difference: {abs(post_backward_memory - estimated_optimizer_memory) / estimated_optimizer_memory * 100:.2f}%")
    print("-" * 80)

    print(f"Projected total memory usage: {convert_size(estimated_model_memory + activation_counter.activation_bytes + estimated_optimizer_memory)}")
    
    print(f"Actual total memory usage: {convert_size(get_current_memory_allocation(device))}")
    print("-" * 80)

