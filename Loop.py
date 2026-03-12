import numpy as np
from cross_attention import cross_attention, d_model
from look_ahead_mask import create_causal_mask, softmax


VOCAB_SIZE = 10_000
SPECIAL_TOKENS = ["<PAD>", "<START>", "<EOS>"]
TARGET_SENTENCE = ["O", "rato", "roeu", "a", "roupa","do","rei","de","Roma","<EOS>"]


def build_mock_vocabulary(vocab_size=VOCAB_SIZE):
	base_tokens = SPECIAL_TOKENS + TARGET_SENTENCE[:-1]
	filler_count = vocab_size - len(base_tokens)
	filler_tokens = [f"tok_{index:05d}" for index in range(filler_count)]
	vocabulary = base_tokens + filler_tokens
	token_to_id = {token: index for index, token in enumerate(vocabulary)}
	return vocabulary, token_to_id

VOCABULARY, TOKEN_TO_ID = build_mock_vocabulary()
OUTPUT_PROJECTION = np.random.default_rng().normal(0.0, 0.02, size=(d_model, VOCAB_SIZE))


def generate_next_token(current_sequence, encoder_out):
	batch_size = encoder_out.shape[0]
	seq_len = len(current_sequence)

	decoder_state = np.random.randn(batch_size, seq_len, d_model)

	causal_mask = create_causal_mask(seq_len)
	
	atencao = decoder_state @ decoder_state.transpose(0, 2, 1) / np.sqrt(d_model)
	mask_atencao = atencao + causal_mask[np.newaxis, :, :]
	decoder_atencao = softmax(mask_atencao) @ decoder_state

	cross_output, _ = cross_attention(encoder_out, decoder_atencao)
	decoder_summary = cross_output[:, -1, :]
	logits = (decoder_summary @ OUTPUT_PROJECTION).reshape(-1)
	base_probabilities = softmax(logits)

	generated_tokens = [token for token in current_sequence if token != "<START>"]
	matched_prefix = 0

	for expected_token, generated_token in zip(TARGET_SENTENCE, generated_tokens):
		if expected_token != generated_token:
			break
		matched_prefix += 1

	next_token = TARGET_SENTENCE[matched_prefix] if matched_prefix < len(TARGET_SENTENCE) else "<EOS>"
	probabilities = base_probabilities * 0.001
	probabilities[TOKEN_TO_ID[next_token]] = 0.999
	probabilities /= np.sum(probabilities)

	return probabilities


def run_inference_loop():
	batch_size = 1
	seq_len_source = 8
	encoder_out = np.random.randn(batch_size, seq_len_source, d_model)

	current_sequence = ["<START>"]

	while True:
		probabilities = generate_next_token(current_sequence, encoder_out)
		next_token_id = int(np.argmax(probabilities))
		next_token = VOCABULARY[next_token_id]
		if next_token == "<EOS>":
			final_sentence = " ".join(token for token in current_sequence if token != "<START>")
			print("Frase final:", final_sentence)
			break

		current_sequence.append(next_token)
		print("Contexto atual:", current_sequence)


if __name__ == "__main__":
	run_inference_loop()
