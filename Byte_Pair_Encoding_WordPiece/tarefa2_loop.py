import re
from tarefa1_frequencias import get_stats, vocab as vocab_inicial

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        new_word = pattern.sub(''.join(pair), word)
        v_out[new_word] = v_in[word]
    return v_out

if __name__ == "__main__":
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
    print("\n[OK] Token 'est</w>' encontrado - Validacao aprovada!\n")
