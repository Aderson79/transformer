# Laboratório 07 — Especialização de LLMs com QLoRA

Pipeline completo de fine-tuning de um modelo de linguagem (OPT-125M) utilizando **QLoRA** (Quantized Low-Rank Adaptation) para o domínio de **suporte técnico de TI**.

## Descrição do Projeto

Este projeto implementa todas as etapas de especialização de um LLM com eficiência de parâmetros:

1. **Geração de dados sintéticos** — 55 pares prompt/response gerados via API da OpenAI (`gpt-4o-mini`), divididos em 90% treino (49) e 10% teste (6), no formato JSONL.
2. **Quantização 4-bit (QLoRA)** — Modelo base carregado em 4-bit NF4 com double quantization via `bitsandbytes`.
3. **Adaptadores LoRA** — Low-Rank Adaptation aplicada nos módulos `q_proj` e `v_proj` (r=64, alpha=16, dropout=0.1).
4. **Treinamento com SFTTrainer** — Otimizador `paged_adamw_32bit`, scheduler cosine, warmup 0.03.
5. **Avaliação** — Comparação qualitativa entre modelo base e modelo fine-tuned nos dados de teste.

### Modelo Base

- **facebook/opt-125m** (125M parâmetros)

### Hiperparâmetros

| Parâmetro | Valor |
|---|---|
| Quantização | 4-bit NF4, double quant |
| LoRA rank (r) | 64 |
| LoRA alpha | 16 |
| LoRA dropout | 0.1 |
| Target modules | q_proj, v_proj |
| Otimizador | paged_adamw_32bit |
| Scheduler | cosine |
| Warmup ratio | 0.03 |
| Learning rate | 2e-4 |
| Batch size | 2 |
| Max steps | 5 |
| Max length | 128 |

## Estrutura do Projeto

```
LoRA_QLoRA/
├── generate_data.py      # Passo 1: Geração de dados sintéticos (OpenAI)
├── train.py              # Passos 2-4: Quantização + LoRA + Treinamento
├── evaluate.py           # Passo 5: Avaliação base vs. fine-tuned
├── colab_train.ipynb     # Notebook para treinamento no Google Colab (GPU)
├── data/
│   ├── train.jsonl       # 49 amostras de treino
│   └── test.jsonl        # 6 amostras de teste
├── outputs/
│   └── qlora-suporte-tecnico/
│       └── final_adapter/ # Adaptadores LoRA treinados
├── context.md            # Especificação do laboratório
├── requirements.txt
├── .env                  # Chave da API OpenAI (não versionado)
└── README.md
```

## Instalação

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar (Windows)
venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

## Uso

### 1. Gerar dados sintéticos

Requer uma chave da API OpenAI no arquivo `.env`:

```
OPENAI_API_KEY=sk-...
```

```bash
python generate_data.py
```

### 2. Treinar o modelo

```bash
python train.py
```

### 3. Avaliar

```bash
python evaluate.py
```

## Dependências

- Python 3.10+
- PyTorch >= 2.1.0
- Transformers >= 4.36.0
- PEFT >= 0.7.0
- BitsAndBytes >= 0.41.0
- TRL >= 0.7.0
- Datasets >= 2.16.0
- Accelerate >= 0.25.0
- OpenAI >= 1.0.0

## Observações sobre Execução

O otimizador `paged_adamw_32bit` (obrigatório conforme especificação do laboratório) requer **GPU com CUDA**. Em ambientes somente CPU, o `bitsandbytes` levanta um erro em `optimizer.step()` pois o paged AdamW utiliza paginação de memória GPU.

O treinamento foi executado com sucesso no **Google Colab** (GPU T4) através do notebook `colab_train.ipynb`, que contém os dados de treino embutidos e configurações otimizadas para GPU (`fp16=True`). Os adaptadores LoRA resultantes estão em `outputs/qlora-suporte-tecnico/final_adapter/`.

## Nota de Integridade

Partes deste projeto foram geradas/complementadas com IA (GitHub Copilot / Claude), revisadas por mim.

### Partes geradas por IA

- **Dados sintéticos** (`data/train.jsonl`, `data/test.jsonl`) — Os 55 pares de pergunta/resposta de suporte técnico foram gerados pela API da OpenAI (`gpt-4o-mini`) através do script `generate_data.py`.
- **Código dos scripts** (`generate_data.py`, `train.py`, `evaluate.py`, `colab_train.ipynb`) — Estrutura e implementação gerados com assistência do GitHub Copilot (Claude), com revisão e ajustes manuais.
- **README.md** — Gerado com assistência de IA, revisado e ajustado manualmente.

