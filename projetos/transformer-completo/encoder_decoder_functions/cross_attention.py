import numpy as np
from encoder_decoder_functions.motorMatematico import scaled_dot_product_attention


def cross_attention(encoder_out, decoder_state, mask=None):
    d_model = decoder_state.shape[-1]

    if not hasattr(cross_attention, "_weight_cache"):
        cross_attention._weight_cache = {}

    if d_model not in cross_attention._weight_cache:
        cross_attention._weight_cache[d_model] = {
            "W_Q": np.random.randn(d_model, d_model),
            "W_K": np.random.randn(d_model, d_model),
            "W_V": np.random.randn(d_model, d_model),
        }

    W_Q = cross_attention._weight_cache[d_model]["W_Q"]
    W_K = cross_attention._weight_cache[d_model]["W_K"]
    W_V = cross_attention._weight_cache[d_model]["W_V"]


    Q = decoder_state @ W_Q      
    K = encoder_out   @ W_K     
    V = encoder_out   @ W_V     

    output, attn_weights = scaled_dot_product_attention(Q, K, V, mask=mask)

    return output, attn_weights






