import numpy as np
import pandas as pd


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)

    if mask is not None:
        scores = scores + mask

    attention_weights = softmax(scores)
    output = attention_weights @ V
    return output, attention_weights


def SelfAttention(X):

    d_model = X.shape[-1]
    d_k = d_model

    if not hasattr(SelfAttention, "_weight_cache"):
        SelfAttention._weight_cache = {}

    if d_model not in SelfAttention._weight_cache:
        SelfAttention._weight_cache[d_model] = {
            "W_Q": np.random.randn(d_model, d_k),
            "W_K": np.random.randn(d_model, d_k),
            "W_V": np.random.randn(d_model, d_k),
        }

    W_Q = SelfAttention._weight_cache[d_model]["W_Q"]
    W_K = SelfAttention._weight_cache[d_model]["W_K"]
    W_V = SelfAttention._weight_cache[d_model]["W_V"]

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    self_attention_output, _ = scaled_dot_product_attention(Q, K, V)

    return self_attention_output


def LayerNorm(X):
    eps=1e-6
    mean = np.mean(X, axis=-1, keepdims=True)
    std = np.std(X, axis=-1, keepdims=True)
    layernorm_output = (X - mean) / (std + eps)
    return layernorm_output

def FFN(X):
    d_ff=4 * X.shape[-1] 
    d_model = X.shape[-1]

    if not hasattr(FFN, "_weight_cache"):
        FFN._weight_cache = {}

    if d_model not in FFN._weight_cache:
        FFN._weight_cache[d_model] = {
            "W1": np.random.randn(d_model, d_ff),
            "b1": np.random.randn(d_ff),
            "W2": np.random.randn(d_ff, d_model),
            "b2": np.random.randn(d_model),
        }

    W1 = FFN._weight_cache[d_model]["W1"]
    b1 = FFN._weight_cache[d_model]["b1"]
    W2 = FFN._weight_cache[d_model]["W2"]
    b2 = FFN._weight_cache[d_model]["b2"]

    aux = np.maximum(0, X @ W1 + b1)  
    ffn_output = aux @ W2 + b2
    return ffn_output