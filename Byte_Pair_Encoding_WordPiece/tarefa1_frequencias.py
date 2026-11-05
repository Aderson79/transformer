from collections import defaultdict

# Vocabulario inicial
vocab = {
    'l o w </w>': 5,
    'l o w e r </w>': 2,
    'n e w e s t </w>': 6,
    'w i d e s t </w>': 3
}

def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


if __name__ == "__main__":
    stats = get_stats(vocab)
    print(f"\nPar ('e', 's'): {stats[('e', 's')]}  (esperado: 9)")
    assert stats[('e', 's')] == 9, "ERRO: par ('e','s') deveria ser 9!"
    print("Validacao aprovada!!!\n")
