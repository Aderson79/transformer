import numpy as np

from decoder_block import DecoderBlock
from encoder_block import encoder_stack

# Tokens especiais e vocabulario base para o modelo.
SPECIAL_TOKENS = ["<PAD>", "<START>", "<EOS>"]
BASE_TOKENS = ["Thinking", "Machines", "are", "the", "future", "of", "AI", "and", "technology"]
VOCAB = SPECIAL_TOKENS + BASE_TOKENS

# Mapas de token para ID e vice-versa.
TOKEN_TO_ID = {token: idx for idx, token in enumerate(VOCAB)}
ID_TO_TOKEN = {idx: token for token, idx in TOKEN_TO_ID.items()}

# IDs dos tokens especiais para facilitar o uso em regras de geracao.
PAD_ID = TOKEN_TO_ID["<PAD>"]
START_ID = TOKEN_TO_ID["<START>"]
EOS_ID = TOKEN_TO_ID["<EOS>"]

#Função criada com ajuda de IA para obter o encoding posicional e a projeção final para o vocabulário.
def build_positional_encoding(seq_len, d_model):
	positions = np.arange(seq_len)[:, np.newaxis]
	dims = np.arange(d_model)[np.newaxis, :]
	angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / np.float32(d_model))
	angle_rads = positions * angle_rates

	positional_encoding = np.zeros((seq_len, d_model))
	positional_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
	positional_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
	return positional_encoding


def embed_tokens(token_ids, embedding_table):
	# Converte IDs em embeddings e soma informacao de posicao.
	seq_len = len(token_ids)
	d_model = embedding_table.shape[1]
	token_embeddings = embedding_table[np.array(token_ids)]
	positional = build_positional_encoding(seq_len, d_model)

	# Neutraliza PAD para nao adicionar conteudo semantico ao contexto.
	pad_positions = np.array(token_ids) == PAD_ID
	token_embeddings[pad_positions] = 0.0
	positional[pad_positions] = 0.0

	return token_embeddings[np.newaxis, :, :] + positional[np.newaxis, :, :]


def transformer_forward(encoder_input, decoder_input, num_encoder_layers=6):
	z = encoder_stack(encoder_input, num_layers=num_encoder_layers)
	decoder_probabilities = DecoderBlock(decoder_input, z, vocab_size=len(VOCAB))
	return decoder_probabilities

# Função criada com ajuda de IA para escolher o próximo token a ser gerado, aplicando regras de controle e penalidades.
def choose_next_token(
	base_probabilities,
	generated_len,
	generated_ids,
	min_tokens_before_eos=2,
	top_k=3,
	repetition_penalty=0.7,
):
	probabilities = np.array(base_probabilities, dtype=np.float64)

	# Evita gerar tokens especiais invalidos no meio da frase.
	probabilities[START_ID] = 0.0
	probabilities[PAD_ID] = 0.0

	# Evita encerrar cedo demais: bloqueia <EOS> nos primeiros passos.
	if generated_len < min_tokens_before_eos:
		probabilities[EOS_ID] = 0.0

	if np.sum(probabilities) <= 0:
		# Fallback para evitar divisao por zero.
		probabilities = np.array(base_probabilities, dtype=np.float64)
		probabilities[START_ID] = 0.0
		probabilities[PAD_ID] = 0.0
		if generated_len < min_tokens_before_eos:
			probabilities[EOS_ID] = 0.0

	# Penalidade de repeticao: reduz chance de tokens ja usados na sequencia.
	for token_id in set(generated_ids):
		if token_id not in (START_ID, PAD_ID):
			probabilities[token_id] *= repetition_penalty

	if np.sum(probabilities) <= 0:
		probabilities = np.array(base_probabilities, dtype=np.float64)
		probabilities[START_ID] = 0.0
		probabilities[PAD_ID] = 0.0
		if generated_len < min_tokens_before_eos:
			probabilities[EOS_ID] = 0.0

	# Top-k: amostra apenas entre os k tokens mais provaveis.
	probabilities /= np.sum(probabilities)
	k = max(1, min(top_k, len(probabilities)))
	top_indices = np.argpartition(probabilities, -k)[-k:]
	top_probs = probabilities[top_indices]
	top_probs /= np.sum(top_probs)
	next_token_id = int(np.random.choice(top_indices, p=top_probs))
	return next_token_id, probabilities


def run_autoregressive_generation():
	d_model = 512
	embedding_table = np.random.randn(len(VOCAB), d_model) * 0.02

	encoder_tokens = ["Thinking", "Machines"]
	encoder_ids = [TOKEN_TO_ID[token] for token in encoder_tokens]
	encoder_input = embed_tokens(encoder_ids, embedding_table)

	print("Frase de entrada do encoder:", " ".join(encoder_tokens))

	generated_ids = [START_ID]
	max_steps = 12

	while True:
		# Embedding dos tokens gerados ate o momento para alimentar o decoder.
		decoder_input = embed_tokens(generated_ids, embedding_table)

		# Passa o contexto do encoder e o input do decoder para obter probabilidades.
		probabilities = transformer_forward(encoder_input, decoder_input)

		# Escolhe o proximo token com base nas probabilidades.
		next_token_id, sampled_probs = choose_next_token(
			probabilities[0, -1, :],
			generated_len=len(generated_ids) - 1,
			generated_ids=generated_ids,
			min_tokens_before_eos=2,
			top_k=3,
			repetition_penalty=0.7,
		)

		# Adiciona o token escolhido a sequencia gerada e imprime o contexto atual.
		generated_ids.append(next_token_id)
		generated_tokens = [ID_TO_TOKEN[idx] for idx in generated_ids]
		print("Contexto atual:", generated_tokens)
		print("Probabilidade do token escolhido:", float(sampled_probs[next_token_id]))

		if next_token_id == EOS_ID:
			break

		# Trava de seguranca para evitar loop infinito.
		if len(generated_ids) >= max_steps:
			generated_ids.append(EOS_ID)
			break

	# Printa a saida final sem tokens de controle.
	final_tokens = [ID_TO_TOKEN[idx] for idx in generated_ids if idx not in (START_ID, PAD_ID)]
	print("Saida final:", " ".join(final_tokens))


if __name__ == "__main__":
	run_autoregressive_generation()
