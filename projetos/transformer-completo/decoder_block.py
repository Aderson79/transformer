import numpy as np
from encoder_decoder_functions.cross_attention import cross_attention
from encoder_decoder_functions.look_ahead_mask import create_causal_mask
from encoder_decoder_functions.motorMatematico import FFN, LayerNorm, scaled_dot_product_attention, softmax


VOCAB_SIZE = 10_000


def _masked_self_attention(y):
	d_model = y.shape[-1]
	seq_len = y.shape[1]

	if not hasattr(_masked_self_attention, "_weight_cache"):
		_masked_self_attention._weight_cache = {}

	if d_model not in _masked_self_attention._weight_cache:
		_masked_self_attention._weight_cache[d_model] = {
			"w_q": np.random.randn(d_model, d_model),
			"w_k": np.random.randn(d_model, d_model),
			"w_v": np.random.randn(d_model, d_model),
		}

	w_q = _masked_self_attention._weight_cache[d_model]["w_q"]
	w_k = _masked_self_attention._weight_cache[d_model]["w_k"]
	w_v = _masked_self_attention._weight_cache[d_model]["w_v"]

	q = y @ w_q
	k = y @ w_k
	v = y @ w_v

	causal_mask = create_causal_mask(seq_len)[np.newaxis, :, :]
	masked_output, _ = scaled_dot_product_attention(q, k, v, mask=causal_mask)
	return masked_output

#Função criada com ajuda de IA para obter as probabilidades 
def _project_to_vocabulary(decoder_output, vocab_size=VOCAB_SIZE):
	d_model = decoder_output.shape[-1]
	cache_key = (d_model, vocab_size)

	if not hasattr(_project_to_vocabulary, "_weight_cache"):
		_project_to_vocabulary._weight_cache = {}

	if cache_key not in _project_to_vocabulary._weight_cache:
		_project_to_vocabulary._weight_cache[cache_key] = np.random.randn(d_model, vocab_size) * 0.02

	output_projection = _project_to_vocabulary._weight_cache[cache_key]
	logits = decoder_output @ output_projection
	probabilities = softmax(logits)
	return probabilities


def decoder_block(y, z, vocab_size=VOCAB_SIZE):
	masked_attention_output = _masked_self_attention(y)
	y = LayerNorm(y + masked_attention_output)
	cross_output, _ = cross_attention(z, y)
	y = LayerNorm(y + cross_output)
	ffn_output = FFN(y)
	y = LayerNorm(y + ffn_output)
# Projeção final para o vocabulário
	probabilities = _project_to_vocabulary(y, vocab_size=vocab_size)
	return probabilities


def DecoderBlock(y, Z, vocab_size=VOCAB_SIZE):
	"""Interface solicitada no enunciado."""
	return decoder_block(y, Z, vocab_size=vocab_size)
