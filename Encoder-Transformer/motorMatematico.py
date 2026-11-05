import numpy as np
import pandas as pd


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def SelfAttention(X):

    d_model = X.shape[-1]
    d_k = d_model

    W_Q = np.random.randn(d_model, d_k)
    W_K = np.random.randn(d_model, d_k)
    W_V = np.random.randn(d_model, d_k)

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
   
    K_T = K.transpose(0, 2, 1) 
    aux = Q @ K_T

    aux_scaled = aux / np.sqrt(d_k)
    
    self_attention_weights = softmax(aux_scaled)
    self_attention_output = self_attention_weights @ V

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
    W1 = np.random.randn(d_model, d_ff)
    b1 = np.random.randn(d_ff)
    W2 = np.random.randn(d_ff, d_model)
    b2 = np.random.randn(d_model)

    aux = np.maximum(0, X @ W1 + b1)  
    ffn_output = aux @ W2 + b2
    return ffn_output