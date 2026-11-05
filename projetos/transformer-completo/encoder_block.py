import numpy as np
from encoder_decoder_functions.motorMatematico import FFN, LayerNorm, SelfAttention


def encoder_block(x):
    attention_output = SelfAttention(x)
    x = LayerNorm(x + attention_output)
    ffn_output = FFN(x)
    x = LayerNorm(x + ffn_output)
    return x


def encoder_stack(x, num_layers=6):
    z = x
    for _ in range(num_layers):
        z = encoder_block(z)
    return z


