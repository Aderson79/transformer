import json
import torch
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

DATA_DIR = Path(__file__).parent / "data"
ADAPTER_DIR = Path(__file__).parent / "outputs" / "qlora-suporte-tecnico" / "final_adapter"
MODEL_NAME = "facebook/opt-125m"


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_prompt(user_question: str) -> str:
    return (
        "Sistema: Você é um assistente de suporte técnico de TI. "
        "Responda de forma clara e profissional.\n"
        f"Usuário: {user_question}\n"
        "Assistente:"
    )


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=False)
    if "Assistente:" in decoded:
        response = decoded.split("Assistente:")[-1].strip()
        return response
    return decoded[len(prompt):].strip()


def main():
    print("=" * 70)
    print("Avaliação: Modelo Base vs. Modelo Fine-Tuned (QLoRA)")
    print("=" * 70)

    test_data = load_jsonl(DATA_DIR / "test.jsonl")
    print(f"\nAmostras de teste: {len(test_data)}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("\n[1/4] Carregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("[2/4] Carregando modelo base (4-bit)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.eval()

    print("[3/4] Gerando respostas do modelo BASE...\n")
    base_responses = []
    for i, sample in enumerate(test_data):
        prompt = build_prompt(sample["prompt"])
        response = generate_response(base_model, tokenizer, prompt)
        base_responses.append(response)
        print(f"  Base [{i+1}/{len(test_data)}] concluído")

    print("\n[4/4] Carregando adaptadores LoRA e gerando respostas FINE-TUNED...\n")
    ft_model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    ft_model.eval()

    ft_responses = []
    for i, sample in enumerate(test_data):
        prompt = build_prompt(sample["prompt"])
        response = generate_response(ft_model, tokenizer, prompt)
        ft_responses.append(response)
        print(f"  Fine-tuned [{i+1}/{len(test_data)}] concluído")

    print("\n" + "=" * 70)
    print("RESULTADOS DA COMPARAÇÃO")
    print("=" * 70)

    results = []
    for i, sample in enumerate(test_data):
        print(f"\n{'─' * 70}")
        print(f"PERGUNTA {i+1}: {sample['prompt']}")
        print(f"{'─' * 70}")
        print(f"\n📋 RESPOSTA ESPERADA:\n   {sample['response']}")
        print(f"\n🔵 MODELO BASE:\n   {base_responses[i]}")
        print(f"\n🟢 MODELO FINE-TUNED:\n   {ft_responses[i]}")

        results.append({
            "pergunta": sample["prompt"],
            "resposta_esperada": sample["response"],
            "resposta_base": base_responses[i],
            "resposta_finetuned": ft_responses[i],
        })

    results_path = Path(__file__).parent / "outputs" / "evaluation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n\nResultados salvos em: {results_path}")
    print("Avaliação concluída!")


if __name__ == "__main__":
    main()
