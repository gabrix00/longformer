import torch
import torch.nn.functional as F
import numpy as np
import math
import spacy
import matplotlib.pyplot as plt

def from_parser2masking(sentence:str, viz:bool = False):
    # Carica il modello linguistico
    nlp = spacy.load("en_core_web_sm")

    sentence = nlp(sentence)
    
    num_tokens = len(sentence)
    mask = np.full((num_tokens, num_tokens), 0) 

    for i, token in enumerate(sentence):
        for j, other_token in enumerate(sentence):
            if i != j:
                if other_token in token.children:
                    mask[i][j] = 1
            else:
                mask[i][i] = 1

    if viz:
        plt.imshow(mask, cmap='Blues', interpolation='nearest')
        plt.title('Attention Mask')
        plt.xticks(range(len([el for  el in sentence])), [str(token) for token in sentence] , rotation=45)
        plt.yticks(range(len([el for  el in sentence])), [str(token) for token in sentence] , rotation=45)
        plt.colorbar()
        plt.show()

    return torch.tensor(mask)


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    contextual_embeddings = torch.matmul(attention, v)
    return contextual_embeddings, attention

seq_len, d_k = 8, 2
q = torch.randn(seq_len, d_k)
k = torch.randn(seq_len, d_k)
v = torch.randn(seq_len, d_k)

#mask = torch.tensor(np.random.choice([0, 1], size=(seq_len, seq_len)))
mask = from_parser2masking("life is a journey, not a destination",viz= True)

contextual_embeddings, attention = scaled_dot_product(q, k, v, mask=mask)

print('\n\n')
print("Mask\n",mask)
print("Q\n", q)
print("K\n", k)
print("V\n", v)
print("Contextual_embeddings\n", contextual_embeddings)
print("Attention\n", attention)
print('\n\n')


