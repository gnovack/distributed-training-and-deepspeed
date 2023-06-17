import torch
from model.transformer import TransformerBlock
from transformers.models.opt.modeling_opt import OPTConfig

from estimate_nn_memory import get_model_memory, ActivationCounter, register_hooks_recursive
from util import format_size, get_device


def get_current_memory_allocation(device):
    if device == "mps":
        return torch.mps.current_allocated_memory()
    elif device == "cuda":
        return torch.cuda.memory_allocated()
    else:
        raise ValueError(f"Unsupported device: {device}")


def get_optimizer_memory(model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Returns the memory usage (in bytes) of the given optimizer and model.
    Note: Currently only supports SGD, Adam, and AdamW.
    """
    model_parameters = sum(param.numel() for param in model.parameters())
    bytes_per_param = 0

    if type(optimizer) == torch.optim.SGD:
        has_momentum = any(param_group.get('momentum', 0) != 0 
                               for param_group in optimizer.param_groups)
        if has_momentum:
            bytes_per_param = 4
    elif type(optimizer) in (torch.optim.Adam, torch.optim.AdamW):
        bytes_per_param = 8
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    
    return model_parameters * bytes_per_param


def project_transformer_memory(
        layers, hidden_size, num_attention_heads, 
        batch_size, sequence_length, optimizer):

    model_memory = 4 * layers * hidden_size * (13 + 12 * hidden_size)

    gradient_memory = model_memory

    # activation memory formula from: https://arxiv.org/pdf/2205.05198.pdf
    activation_memory = layers * batch_size * sequence_length * hidden_size * (
        67 + (9*num_attention_heads*sequence_length) / hidden_size
    )

    optimizer_memory = get_optimizer_memory(model, optimizer)

    return model_memory + gradient_memory + activation_memory + optimizer_memory


if __name__ == "__main__":

    batch_size = 4

    config = OPTConfig(**{
        "hidden_size": 9216,
        "num_attention_heads": 72,
        "enable_bias": True,
        "max_position_embeddings": 512,
        "ffn_dim": 36864,
        "dropout": 0.1
    })
    model = TransformerBlock(config)

    projected_total_memory = format_size(project_transformer_memory(
        1, config.hidden_size,
        config.num_attention_heads, batch_size, 
        config.max_position_embeddings, torch.optim.Adam(model.parameters()))
    )
    print(f"Projected total memory usage: {projected_total_memory}")
    print("-" * 80)


    ############################################################
    ## Measure model memory
    ############################################################
    device = get_device()
    model.to(device)
    
    memory_allocation_with_model = get_current_memory_allocation(device)
    estimated_model_memory = get_model_memory(model)
    
    print(f"Measured Model Memory: {format_size(memory_allocation_with_model)}")
    print(f"Estimated Model Memory: {format_size(estimated_model_memory)}")
    print(f"Percent difference: {abs(memory_allocation_with_model - estimated_model_memory) / estimated_model_memory * 100:.2f}%")
    print("-" * 80)

    ############################################################
    ## Measure activation memory
    ############################################################
    activation_counter = ActivationCounter()
    register_hooks_recursive(model, activation_counter)

    inputs = torch.randn(batch_size, config.max_position_embeddings, config.hidden_size).to(device)
    outputs = model(inputs)

    activation_counter.add_activations(inputs)
    memory_allocation_forward_pass = get_current_memory_allocation(device) - memory_allocation_with_model
    
    print(f"Consumed Activation Memory: {format_size(memory_allocation_forward_pass)}")
    print(f"Estimated Activation Memory: {format_size(activation_counter.activation_bytes)}")
    print(f"Percent difference: {abs(memory_allocation_forward_pass - activation_counter.activation_bytes) / activation_counter.activation_bytes * 100:.2f}%")
    print("-" * 80)

    ############################################################
    ## Measure gradient memory
    ############################################################
    loss_fn = torch.nn.MSELoss()
    labels = torch.randn_like(outputs).to(device)
    labels_size = labels.numel() * labels.element_size()
    outputs_size = outputs.numel() * outputs.element_size()

    loss = loss_fn(outputs, labels)
    loss.backward(retain_graph=True)

    memory_allocation_with_gradients = get_current_memory_allocation(device) - memory_allocation_with_model - memory_allocation_forward_pass
    estimated_gradient_memory = estimated_model_memory

    print(f"Consumed Gradient Memory: {format_size(memory_allocation_with_gradients)}" )
    print(f"Estimated Gradient Memory: {format_size(estimated_gradient_memory)}")
    print(f"Percent difference: {abs(memory_allocation_with_gradients - estimated_gradient_memory) / (estimated_gradient_memory) * 100:.2f}%")
    print("-" * 80)

    ############################################################
    ## Measure optimizer memory
    ############################################################
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.step()

    post_step_memory = get_current_memory_allocation(device) - memory_allocation_with_model - memory_allocation_forward_pass - memory_allocation_with_gradients
    estimated_optimizer_memory = get_optimizer_memory(model, optimizer)

    print(f"Consumed Optimizer + Gradient Memory: {format_size(post_step_memory)}" )
    print(f"Estimated Optimizer + Gradient Memory: {format_size(estimated_optimizer_memory)}")
    print(f"Percent difference: {abs(post_step_memory - estimated_optimizer_memory) / (estimated_optimizer_memory) * 100:.2f}%")
    print("-" * 80)

    print(f"Actual total memory usage: {format_size(get_current_memory_allocation(device))}")

