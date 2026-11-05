import numpy as np
from prepDados import X
from motorMatematico import SelfAttention,LayerNorm,FFN


X_entrada = X.copy()

for layer in range(1, 7):
    X_att = SelfAttention(X)
    X_Nomr1 = LayerNorm(X + X_att)
    X_ffn = FFN(X_Nomr1)
    X_out = LayerNorm(X_Nomr1+ X_ffn)
    X = X_out

if X.shape == X_entrada.shape and X.shape[-1] == 512:
    print(f" Dimensões de X mantidas: {X.shape}")
else:
    raise ValueError(f" Erro: Dimensões de X foram alteradas")

valores_alterados = not np.allclose(X, X_entrada)
assert valores_alterados, " Erro: Os valores de X não foram alterados"

print(f"\n Pesos foram contextualizados: Vetor Z (Z = X modificado com contexto)")
print(f"\n VALIDAÇÃO DE SANIDADE: PASSOU EM TODAS AS VERIFICAÇÕES")

Z = X  #Vetor contextualizado
