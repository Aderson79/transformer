# Laboratório 05 — Treinamento Fim-a-Fim do Transformer (EN → DE)

Este repositório contém a conclusão da **Unidade I**, com a **implementação, integração e treinamento** de uma arquitetura **Transformer** para **tradução de texto (Inglês → Alemão)** usando o dataset **multi30k**.

O foco do laboratório é evoluir os componentes matemáticos do **Laboratório 04** (isolados e testados com pesos estáticos/aleatórios) para um **pipeline completo de treinamento**, onde o modelo **aprende via Backpropagation** com o otimizador **Adam**.

---

## Visão Geral

Neste laboratório, o Transformer é treinado de ponta a ponta, cobrindo:

- Pipeline de dados (download, tokenização e criação de tensores)
- Forward pass Encoder/Decoder com máscaras apropriadas
- Treinamento por múltiplas épocas com otimização
- Inferência auto-regressiva (token a token) após o treino

---

## Mudanças e Adaptações (Lab 04 → Lab 05)

Para atender aos requisitos de treinamento, as principais alterações foram:

### 1) Migração para PyTorch
As funções em NumPy foram refatoradas para **classes que herdam de `torch.nn.Module`**, permitindo:

- Cálculo automático de gradientes (**autograd**)
- Treinamento em GPU (quando disponível)
- Organização modular do modelo

### 2) Gerenciamento de Pesos
O sistema manual de pesos (ex.: dicionários tipo `_weight_cache`) foi substituído por camadas do PyTorch, como:

- `nn.Linear`
- `nn.Embedding`

Isso garante parâmetros treináveis e integração direta com o otimizador.

### 3) Máscara de Atenção Causal (Look-ahead mask)
A máscara de “olhar para frente” foi incorporada ao **forward do Decoder**, garantindo que o modelo **não acesse tokens futuros** durante o treinamento (comportamento auto-regressivo correto).

### 4) Loop de Inferência Auto-regressivo
A inferência foi adaptada para gerar a tradução **token a token**, usando a saída anterior como entrada do próximo passo.

---

## Créditos e Ferramentas de IA

Seguindo as diretrizes do contrato pedagógico, Inteligência Artificial (Gemini/ChatGPT) foi utilizada como **suporte** em:

- **Preparação de dados e tokenização**: integração com `datasets` (Hugging Face) e configuração do `AutoTokenizer` (modelo `bert-base-multilingual-cased`), agilizando a conversão de texto bruto para tensores.
- **Adaptação para PyTorch**: apoio consultivo para converter operações matriciais do NumPy em operações equivalentes com tensores PyTorch, preservando o fluxo de gradientes.

> Observação: a IA foi utilizada como ferramenta de apoio (consultiva), e o entendimento/validação do funcionamento do pipeline e do modelo permaneceu responsabilidade do autor do laboratório.

---

## Requisitos de Sistema

- **Python 3.8+**
- Dependências:

```bash
pip install torch datasets transformers numpy
```

---

## Como Rodar

1. Clone o repositório e garanta que todos os arquivos `.py` estejam na mesma pasta.
2. Execute o script principal:

```bash
python inferencia.py
```

---

## O que esperar ao executar

Ao rodar o script:

- O dataset **multi30k** e o tokenizador serão baixados automaticamente (quando necessário).
- Será iniciado um treinamento de **20 épocas** em um subconjunto de **1000 frases**.
- Você verá a **loss** diminuir a cada época (tendência de convergência).
- Ao final, o modelo tentará gerar uma tradução de uma frase do conjunto de treino para validar a integridade estrutural do pipeline.

---
