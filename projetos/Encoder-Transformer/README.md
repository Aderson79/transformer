# Encoder Transformer

Projeto simples em Python para simular partes de um encoder Transformer:
- preparação de dados (`prepDados.py`)
- operações matemáticas (Self-Attention, LayerNorm e FFN) (`motorMatematico.py`)
- execução do pipeline (`enconder.py`)

## Pré-requisitos

- Python 3.10+ (recomendado)
- `pip`

## Instalação de dependências

No terminal, dentro da pasta do projeto:

```bash
pip install numpy pandas
```

## Como rodar

1. Entre na pasta do projeto:

```bash
cd C:\Users\mesky\Documents\LAB_IA
```

2. Execute o script principal:

```bash
python enconder.py
```

## Saída esperada

Ao rodar corretamente, você deve ver mensagens como:
- `Dimensões de X mantidas: ...`
- `Pesos foram contextualizados: Vetor Z ...`
- `VALIDAÇÃO DE SANIDADE: PASSOU EM TODAS AS VERIFICAÇÕES`

## Estrutura dos arquivos

- `enconder.py`: pipeline principal com 6 camadas de processamento.
- `prepDados.py`: cria os embeddings de entrada (`X`).
- `motorMatematico.py`: implementa `SelfAttention`, `LayerNorm` e `FFN`.

## Observação

Os pesos são inicializados aleatoriamente a cada execução (`np.random.randn`), então os valores numéricos mudam de uma execução para outra.
