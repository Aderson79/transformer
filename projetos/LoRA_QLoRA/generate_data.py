import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from sklearn.model_selection import train_test_split

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = (
    "Você é um gerador de dados de treinamento para um chatbot de suporte técnico de TI. "
    "Gere pares de pergunta (prompt) e resposta (response) realistas sobre problemas comuns "
    "de suporte técnico como: problemas de rede/Wi-Fi, lentidão de computador, erros de software, "
    "problemas com impressora, recuperação de senha, configuração de e-mail, vírus/malware, "
    "backup de dados, problemas com VPN, atualizações de sistema operacional, etc. "
    "As respostas devem ser profissionais, claras e com passos práticos de resolução."
)

GENERATION_PROMPT = (
    "Gere exatamente {n} pares de prompt/response de suporte técnico de TI em formato JSON. "
    "Cada par deve ter um campo 'prompt' (a pergunta/problema do usuário) e 'response' "
    "(a resposta do técnico de suporte). Varie os temas entre: rede, hardware, software, "
    "segurança, e-mail, impressora, VPN, backup, sistema operacional e periféricos. "
    "Retorne APENAS um array JSON válido, sem texto adicional.\n\n"
    "Exemplo de formato:\n"
    '[{{"prompt": "Meu computador está muito lento...", "response": "Existem algumas causas comuns..."}}]'
)

NUM_SAMPLES = 55
BATCH_SIZE = 10
TEST_RATIO = 0.10


def generate_batch(n: int) -> list[dict]:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": GENERATION_PROMPT.format(n=n)},
        ],
        temperature=0.9,
        max_tokens=4096,
    )
    content = response.choices[0].message.content.strip()

    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        content = content.rsplit("```", 1)[0]

    return json.loads(content)


def main():
    all_samples = []
    generated = 0

    print(f"Gerando {NUM_SAMPLES} amostras de suporte técnico...")
    while generated < NUM_SAMPLES:
        batch_n = min(BATCH_SIZE, NUM_SAMPLES - generated)
        print(f"  Lote {generated // BATCH_SIZE + 1}: gerando {batch_n} amostras...")

        try:
            batch = generate_batch(batch_n)
            all_samples.extend(batch)
            generated += len(batch)
            print(f"  -> {len(batch)} amostras geradas (total: {generated})")
        except Exception as e:
            print(f"  Erro no lote: {e}. Tentando novamente em 5s...")
            time.sleep(5)
            continue

        time.sleep(1)

    print(f"\nTotal de amostras geradas: {len(all_samples)}")

    train_data, test_data = train_test_split(
        all_samples, test_size=TEST_RATIO, random_state=42
    )

    print(f"Split: {len(train_data)} treino / {len(test_data)} teste")

    train_path = DATA_DIR / "train.jsonl"
    test_path = DATA_DIR / "test.jsonl"
    all_path = DATA_DIR / "all_samples.jsonl"

    for path, data in [(train_path, train_data), (test_path, test_data), (all_path, all_samples)]:
        with open(path, "w", encoding="utf-8") as f:
            for sample in data:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"Salvo: {path} ({len(data)} amostras)")

    print("\nExemplo de amostra:")
    print(json.dumps(all_samples[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
