import json
import torch
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "qlora-suporte-tecnico"

MODEL_NAME = "facebook/opt-125m"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
    bias="none",
)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def format_chat(sample: dict) -> str:
    return (
        f"Sistema: Você é um assistente de suporte técnico de TI. "
        f"Responda de forma clara e profissional.\n"
        f"Usuário: {sample['prompt']}\n"
        f"Assistente: {sample['response']}"
    )


def main():
    print("=" * 60)
    print("QLoRA Fine-Tuning - Suporte Técnico")
    print("=" * 60)

    print("\n[1/5] Carregando dados de treino...")
    train_data = load_jsonl(DATA_DIR / "train.jsonl")
    train_data = [{"text": format_chat(sample)} for sample in train_data]
    train_dataset = Dataset.from_list(train_data)
    print(f"  Amostras de treino: {len(train_dataset)}")

    print("\n[2/5] Carregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("\n[3/5] Carregando modelo com quantização 4-bit (QLoRA)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    print("\n[4/5] Aplicando adaptadores LoRA...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\n[5/5] Iniciando treinamento...")

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        max_steps=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        learning_rate=2e-4,
        fp16=False,
        use_cpu=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        max_grad_norm=0.3,
        max_length=128,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        args=training_args,
    )

    trainer.train()

    adapter_path = OUTPUT_DIR / "final_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\nAdaptadores LoRA salvos em: {adapter_path}")
    print("Treinamento concluído!")


if __name__ == "__main__":
    main()
