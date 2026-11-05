from transformers import AutoTokenizer

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    frase = "Os hiper-parametros do transformer sao inconstitucionalmente dificeis de ajustar."
    tokens = tokenizer.tokenize(frase)

    print(f"Frase: {frase}")
    print(f"Tokens: {tokens}")
