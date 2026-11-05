import numpy as np
from look_ahead_mask import softmax

batch_size = 1
seq_len_frances = 10
d_model = 512

encoder_output = np.random.randn(batch_size, seq_len_frances, d_model)

seq_len_ingles = 4
decoder_state = np.random.randn(batch_size, seq_len_ingles, d_model)


W_Q = np.random.randn(d_model, d_model)
W_K = np.random.randn(d_model, d_model)
W_V = np.random.randn(d_model, d_model)


def cross_attention(encoder_out, decoder_state):

    Q = decoder_state @ W_Q      
    K = encoder_out   @ W_K     
    V = encoder_out   @ W_V     

    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_model)

    attn_weights = softmax(scores)

    output = attn_weights @ V    

    return output, attn_weights


output, attn_weights = cross_attention(encoder_output, decoder_state)





