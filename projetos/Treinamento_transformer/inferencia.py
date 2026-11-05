from hiperparametros import *
from treinamento import train_model

def test_overfitting(model, tokenizer, sentence_en):
    model.eval()
    print(f"\nTestando Tradução (Overfitting): {sentence_en}")
    
    src_ids = tokenizer.encode(sentence_en, return_tensors='pt').to(DEVICE)
    generated_ids = [tokenizer.cls_token_id] 
    
    for _ in range(MAX_LEN):
        trg_tensor = torch.LongTensor([generated_ids]).to(DEVICE)
        
        with torch.no_grad():
            output = model(src_ids, trg_tensor, tokenizer.pad_token_id)
            
        next_token_id = output[0, -1, :].argmax().item()
        generated_ids.append(next_token_id)
        
        if next_token_id == tokenizer.sep_token_id: 
            break
            
    translation = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"Resultado do Modelo: {translation}")


if __name__ == "__main__":
    trained_model, tokenizer = train_model()
    
    test_sentence = "Two young, white males are outside near many bushes."
    test_overfitting(trained_model, tokenizer, test_sentence)