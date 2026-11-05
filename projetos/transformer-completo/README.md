# Transformer Completo (Projeto Didatico)

Este repositório implementa um fluxo simplificado de Transformer com:

- `Encoder` e `Decoder` em NumPy.
- máscara causal no decoder.
- geração auto-regressiva com token `<START>` até `<EOS>`.
- estratégias de decodificação para reduzir repetição.

## Creditos de IA

As partes abaixo foram criadas ou refinadas com ajuda de IA, conforme comentários no código:

- `transformer.py`
- `build_positional_encoding(...)`.
- `choose_next_token(...)` (regras de controle de geração).

- `decoder_block.py`
- `_project_to_vocabulary(...)` (projeção final + softmax).

Além disso, as seguintes decisões técnicas foram aplicadas como sugestão de IA para melhorar estabilidade e qualidade da geração:

- uso de pesos fixos por cache (em vez de reamostrar a cada chamada).
- uso de `top-k` na seleção de próximo token.
- uso de penalidade de repetição para reduzir loops do mesmo token.

## Justificativa das escolhas

### 1) Pesos fixos (cache)

Quando os pesos são reamostrados a cada passo, cada iteração usa um "modelo diferente" e a sequência fica inconsistente.

Com cache de pesos:

- o mesmo conjunto de pesos é reutilizado durante a execução.
- a decodificação fica mais estável e coerente com o conceito de inferência.
- fica mais fácil depurar e comparar resultados.

### 2) Top-k

Em vez de sempre pegar o maior valor (`argmax`), o `top-k` limita a escolha aos `k` tokens mais prováveis e amostra entre eles.

Benefícios:

- reduz colapso em um único token repetido.
- preserva tokens com boa probabilidade.
- aumenta diversidade sem perder controle.

### 3) Penalidade de repetição

Tokens já gerados têm probabilidade reduzida por um fator (`repetition_penalty`).

Benefícios:

- diminui padrões como `AI AI AI AI ...`.
- melhora variedade local da frase.
- funciona em conjunto com `top-k`.

## Guia de uso (`transformer.py`)

## Pre-requisitos

- Python 3.10+ (recomendado).
- ambiente virtual com NumPy instalado.

## Executar

No PowerShell, na pasta do projeto:

```powershell
& .\.venv\Scripts\Activate.ps1
& .\.venv\Scripts\python.exe .\transformer.py
```

Saída esperada (exemplo):

- `Frase de entrada do encoder: Thinking Machines`
- várias linhas de `Contexto atual: [...]`
- `Saida final: ... <EOS>`

## Ajustar comportamento da geração

No arquivo `transformer.py`, função `choose_next_token(...)`:

- `top_k`: controla diversidade.
- valor menor: mais conservador.
- valor maior: mais variado.

- `repetition_penalty`: controla repetição.
- menor que `1.0`: penaliza mais repetição.
- próximo de `1.0`: penaliza menos.

- `min_tokens_before_eos`: evita encerrar cedo demais.

Exemplo prático:

- `top_k=3`, `repetition_penalty=0.7`: equilíbrio inicial.
- `top_k=5`, `repetition_penalty=0.6`: mais diversidade, menos repetição.

## Estrutura principal

- `transformer.py`: pipeline de inferência auto-regressiva.
- `encoder_block.py`: encoder block e encoder stack.
- `decoder_block.py`: decoder block, máscara causal e projeção para vocabulário.
- `encoder_decoder_functions/motorMatematico.py`: softmax, atenção, layer norm e FFN.
- `encoder_decoder_functions/cross_attention.py`: ponte de cross-attention.

## Observação importante

Este projeto é didático e não representa um modelo treinado em corpus real.

Por isso:

- mesmo com melhorias de decodificação, as frases podem não ser semanticamente ideais.
- para qualidade real, é necessário treino supervisionado de parâmetros do modelo.
