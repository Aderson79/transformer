import pandas as pd
import numpy as np

vocab=["o","banco","bloqueou","cartao"]
id=[0,1,2,3]

df=pd.DataFrame({"id":id,"palavras":vocab})

#Frase de entrada: "o banco bloqueou o cartao"
#Frase -> Lista de token
palavraEntrada=[0,1,2,0,3]

d_model=512
vocab_size=len(vocab)

embedding_table = np.random.randn(vocab_size, d_model)
embeddings = embedding_table[palavraEntrada]        


X = np.expand_dims(embeddings, axis=0)                




