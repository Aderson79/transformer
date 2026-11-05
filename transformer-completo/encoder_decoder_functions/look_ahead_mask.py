import numpy as np
from encoder_decoder_functions.motorMatematico import softmax


def create_causal_mask(seq_len):

    triangulo_inferior = np.tril(np.ones((seq_len, seq_len)), k=0)

    mask = np.where(triangulo_inferior == 1, 0, -np.inf)

    return mask

