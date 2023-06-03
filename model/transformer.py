import torch
from torch import nn
from transformers import OPTConfig, OPTModel


class MatMul(nn.Module):
    """
    PyTorch Module wrapper for torch.bmm.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.bmm(x, y)


class TransformerBlock(nn.Module):
    """Minimal transformer implementation for memory estimation.
    
    Every operation is implemented as an instance of nn.Module to enable
    accurate estimation of activation memory via forward hooks.

    This implementation is based on the OPTDecoder class in `transformers`:
    https://github.com/huggingface/transformers/blob/v4.29.2/src/transformers/models/opt/modeling_opt.py#L481
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.ffn_dim = config.ffn_dim
        self.scaling = self.head_dim**-0.5
        
        self.pre_attention_layer_norm = nn.LayerNorm(self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.enable_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.enable_bias)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.enable_bias)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.enable_bias)

        self.compute_attention_weights = MatMul()
        self.attention_weights_softmax = nn.Softmax(dim=-1)
        self.attention_weights_dropout = nn.Dropout(config.dropout)
        self.compute_attentions = MatMul()

        self.attention_output_dropout = nn.Dropout(config.dropout)
        self.final_layer_norm = nn.LayerNorm(self.hidden_size)

        self.ffn_1 = nn.Linear(self.hidden_size, self.ffn_dim, bias=config.enable_bias)
        self.ffn_2 = nn.Linear(self.ffn_dim, self.hidden_size, bias=config.enable_bias)
        self.activation = nn.ReLU()
        self.final_dropout = nn.Dropout(config.dropout)
        

    def forward(self, hidden_states):
        residual = hidden_states

        batch_size, target_length, _ = hidden_states.size()

        hidden_states = self.pre_attention_layer_norm(hidden_states)
        
        query = (self.q_proj(hidden_states) * self.scaling).view(
            batch_size, target_length, self.num_attention_heads, self.head_dim
        ).transpose(1, 2).contiguous()
        
        key = self.k_proj(hidden_states).view(
            batch_size, -1, self.num_attention_heads, self.head_dim
        ).transpose(1, 2).contiguous()

        value = self.v_proj(hidden_states).view(
            batch_size, -1, self.num_attention_heads, self.head_dim
        ).transpose(1, 2).contiguous()

        projection_shape = (batch_size * self.num_attention_heads, -1, self.head_dim)
        query = query.view(*projection_shape)
        key = key.view(*projection_shape)
        value = value.view(*projection_shape)

        attention_weights = self.compute_attention_weights(query, key.transpose(1, 2))
        attention_weights = self.attention_weights_softmax(attention_weights)
        attention_weights = self.attention_weights_dropout(attention_weights)

        attention_output = self.compute_attentions(attention_weights, value).view(
            batch_size, self.num_attention_heads, target_length, self.head_dim
        ).transpose(1, 2)

        attention_output = attention_output.reshape(batch_size, target_length, self.hidden_size)
        attention_output = self.out_proj(attention_output)

        attention_output = self.attention_output_dropout(attention_output)
        hidden_states = attention_output + residual

        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.ffn_1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.ffn_2(hidden_states)
        hidden_states = self.final_dropout(hidden_states)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)
        return hidden_states



def _block_from_pretrained():
    pretrained_decoder = OPTModel.from_pretrained("facebook/opt-125m").decoder.layers[0]
    
    model = TransformerBlock(OPTConfig.from_pretrained("facebook/opt-125m"))

    model.pre_attention_layer_norm.load_state_dict(pretrained_decoder.self_attn_layer_norm.state_dict())
    model.k_proj.load_state_dict(pretrained_decoder.self_attn.k_proj.state_dict())
    model.q_proj.load_state_dict(pretrained_decoder.self_attn.q_proj.state_dict())
    model.v_proj.load_state_dict(pretrained_decoder.self_attn.v_proj.state_dict())
    model.out_proj.load_state_dict(pretrained_decoder.self_attn.out_proj.state_dict())

    model.final_layer_norm.load_state_dict(pretrained_decoder.final_layer_norm.state_dict())
    model.ffn_1.load_state_dict(pretrained_decoder.fc1.state_dict())
    model.ffn_2.load_state_dict(pretrained_decoder.fc2.state_dict())

    return model


if __name__ == "__main__":
    model = _block_from_pretrained()
    model.eval()

    inputs = torch.randn(1, model.config.max_position_embeddings, model.config.hidden_size)
    outputs = model(inputs)
    
    pretrained_decoder = OPTModel.from_pretrained("facebook/opt-125m").decoder.layers[0]
    pretrained_decoder.eval()
    pretrained_outputs = pretrained_decoder(inputs)

    print(f"Model Outputs: {outputs}")
    print(f"Pretrained Outputs: {pretrained_outputs[0]}")