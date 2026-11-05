# Transformer

Repositório unificado com projetos didáticos sobre a arquitetura Transformer,
tokenização, treinamento e especialização de modelos de linguagem.

Os projetos foram desenvolvidos em etapas independentes e mantidos em suas
próprias subpastas para preservar seus códigos, dependências e históricos.

## Conteúdo

| Projeto | Descrição |
|---|---|
| [`Encoder-Transformer`](./Encoder-Transformer) | Implementação didática de um encoder Transformer usando NumPy, com self-attention, LayerNorm e FFN. |
| [`Decoder`](./Decoder) | Máscara causal, cross-attention e loop de inferência autoregressivo em NumPy. |
| [`transformer-completo`](./transformer-completo) | Pipeline didático de encoder, decoder e geração autoregressiva com top-k e penalidade de repetição. |
| [`Treinamento_transformer`](./Treinamento_transformer) | Transformer treinado de ponta a ponta para tradução Inglês → Alemão usando PyTorch e o dataset Multi30k. |
| [`Byte_Pair_Encoding_WordPiece`](./Byte_Pair_Encoding_WordPiece) | Implementação de BPE do zero e experimentos com WordPiece usando Hugging Face. |
| [`LoRA_QLoRA`](./LoRA_QLoRA) | Fine-tuning de um modelo OPT-125M com quantização 4-bit e adaptadores QLoRA. |
| [`RAG_QLORA_CPU_OTIMIZATION`](./RAG_QLORA_CPU_OTIMIZATION) | Laboratório de otimização e experimentação de RAG/QLoRA em ambiente com recursos limitados. |

## Requisitos gerais

- Python 3.10 ou superior é recomendado.
- `pip` e ambientes virtuais são recomendados.
- Os projetos possuem dependências diferentes. Consulte o `README.md` ou
  `requirements.txt` dentro de cada subpasta antes de executar os scripts.

## Instalação

Clone o repositório e entre no projeto desejado:

```powershell
git clone https://github.com/Aderson79/transformer.git
cd transformer
cd transformer-completo
```

Quando houver um arquivo de dependências:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Alguns projetos sem `requirements.txt` indicam seus pacotes diretamente no
README local. Por exemplo:

```powershell
pip install numpy pandas
```

## Executando os projetos

Os comandos principais são executados dentro da respectiva subpasta:

```powershell
# Encoder
cd Encoder-Transformer
python enconder.py

# Decoder
cd ..\Decoder
python Loop.py

# Transformer completo
cd ..\transformer-completo
python transformer.py

# Tokenização BPE e WordPiece
cd ..\Byte_Pair_Encoding_WordPiece
python bpe_wordpiece.py

# Treinamento e inferência
cd ..\Treinamento_transformer
python inferencia.py
```

Cada projeto possui instruções específicas, parâmetros e exemplos de saída no
seu README.

## Treinamento e hardware

Os projetos de inferência e tokenização podem ser executados localmente com
CPU, desde que suas dependências estejam instaladas.

O treinamento com QLoRA em [`LoRA_QLoRA`](./LoRA_QLoRA) foi configurado para
GPU CUDA e `bitsandbytes`. Para esse projeto, o notebook de Google Colab
contém uma alternativa prática de execução. O treinamento completo pode não
funcionar corretamente em máquinas somente com CPU.

O projeto de treinamento do Transformer pode baixar automaticamente o dataset
Multi30k e o tokenizador na primeira execução.

## Chaves e dados sensíveis

O projeto [`LoRA_QLoRA`](./LoRA_QLoRA) pode usar a API da OpenAI para gerar
dados sintéticos. Nunca inclua chaves reais no Git:

```text
OPENAI_API_KEY=sua-chave-aqui
```

Use um arquivo `.env` local, mantenha-o fora do versionamento e substitua
qualquer chave que tenha sido exposta acidentalmente.

## Organização do repositório

```text
transformer/
├── Encoder-Transformer/
├── Decoder/
├── transformer-completo/
├── Treinamento_transformer/
├── Byte_Pair_Encoding_WordPiece/
├── LoRA_QLoRA/
└── RAG_QLORA_CPU_OTIMIZATION/
```

## Escopo

Este é um repositório de estudos e laboratórios. As implementações têm
objetivo didático e não pretendem substituir frameworks de produção ou
modelos treinados em larga escala. Pesos aleatórios, conjuntos de dados
pequenos e configurações reduzidas podem produzir resultados instáveis ou
apenas demonstrativos.

Consulte o README de cada projeto para detalhes de implementação, limitações,
dependências e créditos de ferramentas de IA utilizadas durante o
desenvolvimento.
