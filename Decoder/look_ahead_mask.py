import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def create_causal_mask(seq_len):

    triangulo_inferior = np.tril(np.ones((seq_len, seq_len)), k=0)

    mask = np.where(triangulo_inferior == 1, 0, -np.inf)

    return mask


def demonstracao():
    seq_len = 5
    
    mask = create_causal_mask(seq_len)
    print(f"Mascara Causal:")
    print(mask)
    
    d_k = 4  
    Q = np.random.randn(seq_len, d_k) * 0.5
    K = np.random.randn(seq_len, d_k) * 0.5
    
    atencao = np.matmul(Q, K.T)
  
    mask_atencao = atencao + mask
    
    softmax_atencao = softmax(mask_atencao)
    
    print(f"\nProbabilidades apos Softmax:")
    print(np.round(softmax_atencao, 4))
    
    return mask, atencao, mask_atencao, softmax_atencao


if __name__ == "__main__":
    mask, scores, mask_atencao, attention = demonstracao()
