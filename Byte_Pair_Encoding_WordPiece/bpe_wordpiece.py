from tarefa1_frequencias import get_stats, vocab as vocab_inicial
from tarefa2_loop import merge_vocab
from transformers import AutoTokenizer

print("\n=== TAREFA 1 ===\n")
stats = get_stats(vocab_inicial)
print(f"Par ('e', 's'): {stats[('e', 's')]}  (esperado: 9)")
assert stats[('e', 's')] == 9, "ERRO: par ('e','s') deveria ser 9!"
print("Validacao aprovada!!!\n")


print("=== TAREFA 2 ===\n")
vocab = dict(vocab_inicial)
num_merges = 5

for i in range(num_merges):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get)
    print(f"Iteracao {i + 1}: merge {best} (freq={pairs[best]})")
    vocab = merge_vocab(best, vocab)

print(f"\nVocabulario final apos {num_merges} merges:")
for word, freq in vocab.items():
    print(f"  {word}: {freq}")

tokens_finais = ' '.join(vocab.keys())
assert 'est</w>' in tokens_finais, "ERRO: token 'est</w>' nao foi criado!"
print("\nToken 'est</w>' encontrado - Validacao aprovada!!!\n")


print("=== TAREFA 3 ===\n")
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

frase = "Os hiper-parametros do transformer sao inconstitucionalmente dificeis de ajustar."
tokens = tokenizer.tokenize(frase)

print(f"Frase: {frase}\n")
print(f"Tokens: {tokens}")
