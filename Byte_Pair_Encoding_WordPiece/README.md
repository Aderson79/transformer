# Laboratório 6 — Tokenização: BPE (do zero) & WordPiece (Hugging Face)

## Descrição

Implementação do algoritmo **Byte Pair Encoding (BPE)** do zero e exploração do **WordPiece** via Hugging Face, utilizando o modelo `bert-base-multilingual-cased`.

## Como executar

```bash
pip install transformers
python bpe_wordpiece.py
```

## Estrutura do projeto

| Arquivo | Descrição |
|---------|-----------|
| `bpe_wordpiece.py` | Arquivo principal — importa e executa todas as tarefas |
| `tarefa1_frequencias.py` | Tarefa 1 — Motor de frequências (`get_stats`) e vocabulário inicial |
| `tarefa2_merges.py` | Tarefa 2 — Função de fusão (`merge_vocab`) e loop BPE |
| `tarefa3_wordpiece.py` | Tarefa 3 — WordPiece com Hugging Face (BERT) |
| `README.md` | Documentação do projeto |

Cada tarefa pode ser executada individualmente (`python tarefa1_frequencias.py`) ou todas juntas via `python bpe_wordpiece.py`.

## Tarefas implementadas

### Tarefa 1 — Motor de Frequências (BPE)

Função `get_stats(vocab)` que percorre o vocabulário e conta a frequência de cada par adjacente de símbolos. Validação: o par `('e', 's')` resulta em **9** (6 de *newest* + 3 de *widest*).

### Tarefa 2 — Loop de Fusão (Merges)

Função `merge_vocab(pair, v_in)` que utiliza regex (com `re.escape`) para substituir o par mais frequente pelo símbolo unido. O loop executa **5 iterações**, produzindo as fusões:

1. `('e', 's')` → `es`
2. `('es', 't')` → `est`
3. `('est', '</w>')` → `est</w>`
4. `('l', 'o')` → `lo`
5. `('lo', 'w')` → `low`

### Tarefa 3 — WordPiece com Hugging Face

Tokenização da frase `"Os hiper-parâmetros do transformer são inconstitucionalmente difíceis de ajustar."` usando `bert-base-multilingual-cased`.

Tokens resultantes:
```
['Os', 'hip', '##er', '-', 'par', '##âm', '##etros', 'do', 'transform', '##er',
 'são', 'in', '##cons', '##tit', '##uc', '##ional', '##mente', 'di', '##f',
 '##í', '##cei', '##s', 'de', 'aj', '##usta', '##r', '.']
```

## O que significa `##` nos tokens?

O prefixo `##` no WordPiece indica que aquele token é uma **continuação** da palavra anterior, ou seja, não é o início de uma nova palavra. Por exemplo:

- `transform` + `##er` = **transformer**
- `in` + `##cons` + `##tit` + `##uc` + `##ional` + `##mente` = **inconstitucionalmente**

O tokenizer divide palavras desconhecidas ou longas em sub-palavras conhecidas do vocabulário. O `##` serve como marcador para que o modelo saiba que aquele fragmento deve ser concatenado ao token anterior durante a reconstrução.

## Por que sub-palavras evitam o problema de OOV (Out-of-Vocabulary)?

Modelos com vocabulário fixo de palavras inteiras travam ao encontrar palavras nunca vistas no treinamento — o chamado problema **OOV**. A tokenização por sub-palavras resolve isso porque:

1. **Decomposição morfológica:** Qualquer palavra desconhecida pode ser decomposta em fragmentos menores que existem no vocabulário. Por exemplo, `inconstitucionalmente` não precisa estar no dicionário inteiro — basta que `in`, `##cons`, `##tit`, `##uc`, `##ional` e `##mente` estejam.

2. **Vocabulário finito cobre infinitas palavras:** Com um conjunto limitado de sub-palavras (geralmente ~30k tokens), o modelo consegue representar qualquer palavra de qualquer idioma, incluindo neologismos, termos técnicos e erros de digitação.

3. **Preservação semântica:** Sub-palavras frequentemente correspondem a morfemas reais (raízes, prefixos, sufixos), permitindo que o modelo aproveite padrões morfológicos compartilhados entre palavras diferentes.

## Citação de IA

As funções `get_stats` e `merge_vocab` foram desenvolvidas com auxílio do GitHub Copilot (Claude) para realizar a manipulação correta das strings. A lógica de regex em `merge_vocab` (uso de `re.escape` e lookbehind/lookahead `(?<!\S)` / `(?!\S)`) foi criada com auxílio pela IA e revisada para garantir que tokens já fundidos não fossem quebrados.
