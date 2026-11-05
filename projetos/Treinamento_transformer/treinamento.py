import torch.nn as nn
import torch.optim as optim
from tokenização import prepare_data
from transformer import LabTransformer
from hiperparametros import DEVICE, D_MODEL, NUM_HEADS, NUM_LAYERS, LR, EPOCHS


def train_model():
    loader, tokenizer = prepare_data()
    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.pad_token_id
    
    model = LabTransformer(vocab_size, D_MODEL, NUM_HEADS, NUM_LAYERS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id) 

    print(f"Iniciando Treinamento em {DEVICE}...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in loader:
            src = batch['src_ids'].to(DEVICE)
            trg = batch['trg_ids'].to(DEVICE)
            
            trg_input = trg[:, :-1]
            trg_output = trg[:, 1:] 
            
            optimizer.zero_grad()
            
            output = model(src, trg_input, pad_id)
            
            loss = criterion(output.reshape(-1, vocab_size), trg_output.reshape(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Época {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f}")
    
    return model, tokenizer