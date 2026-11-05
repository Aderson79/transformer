# Decoder — Simulação do Loop de Inferência

Ess atividade implementa, do zero com NumPy, as peças centrais de um Decoder Transformer:
máscara causal, cross-attention e um loop de inferência autoregressivo com vocabulário fictício.

---

## Arquivos

| Arquivo | Responsabilidade |
|---|---|
| `look_ahead_mask.py` | `softmax` numérico e `create_causal_mask` (máscara triangular inferior) |
| `cross_attention.py` | Matrizes de projeção Q, K, V e função `cross_attention` |
| `Loop.py` | Vocabulário fictício, `generate_next_token` e loop de inferência |

---

## Pré-requisitos

- Python 3.10 ou superior
- NumPy

### Instalação do NumPy

```bash
pip install numpy
```

---

## Como executar

Cada arquivo pode ser rodado de forma independente.

### Máscara causal

```bash
python look_ahead_mask.py
```

Exibe a máscara triangular e as probabilidades de atenção após o softmax.

### Cross-Attention

```bash
python cross_attention.py
```

Inicializa pesos aleatórios e executa uma passagem de cross-attention entre um encoder e um decoder fictícios.

### Loop de inferência

```bash
python Loop.py
```

Executa o loop autoregressivo: a cada passo, o modelo gera o próximo token usando `argmax` sobre o vetor de probabilidades. O loop para ao gerar `<EOS>` e imprime a frase final.

Saída esperada:

```
Contexto atual: ['<START>', 'O']
Contexto atual: ['<START>', 'O', 'rato']
...
Frase final: O rato roeu a roupa do rei de Roma
```

---

## Como o loop de inferência funciona (`Loop.py`)

```
encoder_out  ──────────────────────────────────────────┐
                                                        │
current_sequence                                        │
      │                                                 │
      ▼                                                 │
decoder_state = randn(batch, seq_len, d_model)          │
      │                                                 │
      ▼                                                 │
Self-Attention + Máscara Causal                         │
      │   (look_ahead_mask.create_causal_mask)          │
      │   (look_ahead_mask.softmax)                     │
      ▼                                                 │
decoder_atencao  ◄──────────────────────────── cross_attention(encoder_out, decoder_atencao)
      │
      ▼
decoder_summary = cross_output[:, -1, :]   ← último token
      │
      ▼
logits = decoder_summary @ OUTPUT_PROJECTION   shape: (VOCAB_SIZE,)
      │
      ▼
probabilities = softmax(logits)
      │
      ▼
next_token = VOCABULARY[argmax(probabilities)]
      │
      ├── == "<EOS>"  →  imprime frase final e encerra
      └── != "<EOS>"  →  append em current_sequence e repete
```

---

## Uso de Inteligência Artificial

Durante o desenvolvimento deste projeto, foi utilizada IA (GitHub Copilot) como ferramenta de apoio nos seguintes pontos:

- **Sintaxe do NumPy** — dúvidas sobre operações como `np.newaxis`, `transpose`, `np.tril`, `np.where` e broadcasting entre tensores de dimensões diferentes.

- **Mock Vocabulary** — entendimento de como estruturar um vocabulário fictício com tokens especiais (`<PAD>`, `<START>`, `<EOS>`), tokens reais e tokens de preenchimento (`tok_XXXXX`), e como mapear cada token para um índice numérico com `token_to_id`.

- **Etapas do `generate_next_token`** — compreensão do fluxo interno da função
