import torch
import numpy as np
import spacy
import matplotlib.pyplot as plt

# Carica il modello linguistico
nlp = spacy.load("en_core_web_sm")

# Definisci la frase di esempio
sentence = nlp("life is a journey, not a destination")

# Creazione della matrice  
num_tokens = len(sentence)
matrix = np.full((num_tokens, num_tokens), 0)

# Per ogni token nella frase, stampa il token e le sue dipendenze
for token in sentence:
    print(token.text, [child.text for child in token.children])
   # Extract next neighboring node of `developer`
    #print(token.text, [child.text for child in token.nbor()])

# Ciclo per scorrere la lista dei token nella frase
for i, token in enumerate(sentence):
    # Ciclo per scorrere nuovamente la lista dei token nella frase
    for j, other_token in enumerate(sentence):
        # Controlla se l'indice corrente non è uguale all'indice dell'altro token
        if i != j:
            # Controlla se l'altro token è un figlio del token corrente
            if other_token in token.children:
                # Imposta a 1 la cella corrispondente all'altro token
                matrix[i][j] = 1
        else:
            matrix[i][i] = 1


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.eq(0).unsqueeze(1).expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k
    return pad_attn_mask


# Definizione dei dati di input di esempio
# Supponiamo che input_ids sia una sequenza di token rappresentata come tensori di PyTorch
#input_ids = torch.tensor([[101, 0, 5678, 0, 0], [101, 321, 654, 987, 0]])  # Esempio di due sequenze di token
input_ids = torch.tensor(matrix)
# Chiamata alla funzione get_attn_pad_mask con i dati di input di esempio
pad_mask = get_attn_pad_mask(input_ids, input_ids)

# Stampa della maschera di attenzione per il primo elemento del batch e il primo token
for i in pad_mask:
    print(i[0])
print("Maschera di attenzione per il primo elemento del batch e il primo token:")
print(pad_mask[0][0])
####################################################################
import torch.nn as nn

class SingleHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super(SingleHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k)
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_v)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)

        q_s = self.W_Q(Q)  # q_s: [batch_size x len_q x d_k]
        k_s = self.W_K(K)  # k_s: [batch_size x len_k x d_k]
        v_s = self.W_V(V)  # v_s: [batch_size x len_k x d_v]

        # Calcolare l'attenzione usando il prodotto scalare tra q e k
        attn_score = torch.matmul(q_s, k_s.transpose(-2, -1)) / (K.size(-1) ** 0.5)

        # Applicare la maschera di attenzione
        if attn_mask is not None:
            attn_score.masked_fill_(attn_mask, -1e9)  # Applicare una maschera con un valore molto negativo

        # Calcolare l'attenzione pesata
        attn_weight = nn.functional.softmax(attn_score, dim=-1)
        context = torch.matmul(attn_weight, v_s)  # context: [batch_size x len_q x d_v]

        # Aggiungere la connessione residuale e applicare la normalizzazione layer
        output = nn.LayerNorm(Q.size(-1))(context + residual)

        return output, attn_weight
    
class ScaledDotProductAttentionSingleHead(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttentionSingleHead, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_k ** 0.5)  # scores : [batch_size x len_q x len_k]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return scores, context, attn
#############################################################
# Definizione dei dati di input di esempio
input_ids = torch.tensor(matrix)

# Chiamata alla funzione get_attn_pad_mask con i dati di input di esempio
pad_mask = get_attn_pad_mask(input_ids, input_ids)
print("Maschera di attenzione di padding:")
print(pad_mask)
    
# Definizione della dimensione delle chiavi
d_k = 6
# Creazione di un'istanza della classe ScaledDotProductAttentionSingleHead
attention_layer = ScaledDotProductAttentionSingleHead(d_k)

# Definizione dei tensori per le query, le chiavi e i valori (utilizzando input_ids come valori)
Q = input_ids
K = input_ids
V = input_ids

# Calcolo della lunghezza della sequenza di input
batch_size, len_q = Q.size()
batch_size, len_k = K.size()

# Calcolo della maschera di attenzione
pad_attn_mask = pad_mask

# Chiamata alla funzione forward della classe ScaledDotProductAttentionSingleHead
context, attn = attention_layer(Q, K, V, pad_attn_mask)

print("Contesto:")
print(context)

print("Attenzione:")
print(attn)
