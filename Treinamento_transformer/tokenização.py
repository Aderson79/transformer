from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from hiperparametros import MODEL_NAME, MAX_LEN, BATCH_SIZE

def prepare_data():
    print("Carregando Dataset e Tokenizador...")
    dataset = load_dataset("bentrevett/multi30k", trust_remote_code=True)
    train_data = dataset['train'].select(range(1000)) # Subconjunto de 1000 frases
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_fn(examples):
        # Inserindo <START> e <EOS> conforme PDF
        src_texts = examples['en']
        trg_texts = ["[CLS] " + t + " [SEP]" for t in examples['de']] # BERT usa CLS/SEP como Start/End
        
        src_enc = tokenizer(src_texts, padding='max_length', truncation=True, max_length=MAX_LEN)
        trg_enc = tokenizer(trg_texts, padding='max_length', truncation=True, max_length=MAX_LEN)
        
        return {
            'src_ids': src_enc['input_ids'],
            'trg_ids': trg_enc['input_ids']
        }

    tokenized_dataset = train_data.map(tokenize_fn, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['src_ids', 'trg_ids'])
    
    loader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader, tokenizer