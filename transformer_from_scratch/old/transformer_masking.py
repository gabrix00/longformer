"""
A from scratch implementation of Transformer network,
following the paper Attention is all you need with a
few minor differences. I tried to make it as clear as
possible to understand and also went through the code
on my youtube channel!


"""
import numpy as np
import spacy

import torch.nn.functional as F

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


from transformers import BertTokenizer


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask=None):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)  # (N, value_len, embed_size)
        keys = self.keys(keys)  # (N, key_len, embed_size)
        queries = self.queries(query)  # (N, query_len, embed_size)

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        return out
    
def from_parser2masking(sentence:str):
    # Carica il modello linguistico
    nlp = spacy.load("en_core_web_sm")

    sentence = nlp(sentence)
    #DA FIXARE PROBLMA TOKENIZER BERT CON TOKENIZER SPACY (FORSE CAMBIARE SPACY CON ALTRO MODELLO DI PARSING)
    # Tokenizza il testo e converte in tensori
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens = tokenizer(sentence, return_tensors='pt')

    matrix = np.full((tokens["input_ids"].shape[1], tokens["input_ids"].shape[1]), 0) 

    for i, token in enumerate(np.nditer(tokens["input_ids"])):
        for j, other_token in enumerate(np.nditer(tokens["input_ids"])):
            if i != j:
                # Controlla se l'altro token è un figlio del token corrente
                if other_token in token.children:
                    # Imposta a 1 la cella corrispondente all'altro token
                    matrix[i][j] = 1
            else:
                matrix[i][i]
    
    mask = torch.tensor(matrix)
    return mask


    #return np.vstack([np.vstack([np.ones(l),np.torch.tensor(matrix)]),np.ones(l)]) #aggiungo una dimensione alla fine e uno all'inzio per il token cls e sep
                

def from_parser2masking_temp(sentence:str):
    # Carica il modello linguistico
    nlp = spacy.load("en_core_web_sm")

    sentence = nlp(sentence)
    #DA FIXARE PROBLMA TOKENIZER BERT CON TOKENIZER SPACY (FORSE CAMBIARE SPACY CON ALTRO MODELLO DI PARSING)
    # Tokenizza il testo e converte in tensori
    num_tokens = len(sentence)
    mask = np.full((num_tokens, num_tokens), 0) 


    for i, token in enumerate(sentence):
        for j, other_token in enumerate(sentence):
            if i != j:
                # Controlla se l'altro token è un figlio del token corrente
                if other_token in token.children:
                    # Imposta a 1 la cella corrispondente all'altro token
                    mask[i][j] = 1
            else:
                mask[i][i] = 1

    #mask = np.vstack([np.ones(num_tokens),torch.tensor(mask)])
    #mask = np.vstack([torch.tensor(mask),np.ones(num_tokens)])

    return torch.tensor(mask)

#print(from_parser2masking_temp("life is a journey, not a destination"))


'''
############## VISUALIZZAZIONE ATTENTION MASK ############
import spacy
nlp = spacy.load("en_core_web_sm")

sentence = nlp("CLS life is a journey, not a destination SEP")
num_tokens = len(sentence)
print([el for  el in sentence])


mask = np.full((num_tokens , num_tokens ), 0) 

for i, token in enumerate(sentence):
    for j, other_token in enumerate(sentence):
        # Imposta l'attenzione a 0 se uno dei token è mascherato o se ci sono token duplicati
        if i != j:
            if other_token in token.children:
                    # Imposta a 1 la cella corrispondente all'altro token
                    mask[i][j] = 1
        else:
            mask[i][i] = 1


plt.imshow(mask, cmap='Blues', interpolation='nearest')
plt.title('Attention Mask')
plt.xticks(range(len([el for  el in sentence])), [str(token) for token in sentence] , rotation=45)
plt.yticks(range(len([el for  el in sentence])), [str(token) for token in sentence] , rotation=45)
plt.colorbar()
plt.show()
#####################
'''


'''
#####################################################
# Definisci la frase di esempio
#from_parser2masking("life is a journey, not a destination")
nlp = spacy.load("en_core_web_sm")             
sentence = nlp("life is a journey, not a destination")
print(sentence)
print(len(sentence))
print(type(sentence))
#--------------------------------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer("life is a journey, not a destination", return_tensors='pt')
print(len(tokens["input_ids"][0]))
print(type(tokens))
#--------------------------------------------------
'''
'''
tokens = tokenizer("life is a journey, not a destination", return_tensors='pt')
for i, token in enumerate(np.nditer(tokens["input_ids"])):
    print(f"Indice: {i}, Token: {token}")

#matrix = np.full((tokens["input_ids"].shape[1], tokens["input_ids"].shape[1]),0 )  # Matrice di zeri interi
mask= np.random.choice([0, 1], size=(tokens["input_ids"].shape[1], tokens["input_ids"].shape[1]))
mask = torch.from_numpy(mask)
print(mask)
'''
#mask = from_parser2masking("life is a journey, not a destination")





########

heads = 1
embedding_dim = 10
model = SelfAttention(embedding_dim, heads)

#print(from_parser2masking_temp("CLS life is a journey, not a destination SEP"))
mask = from_parser2masking_temp("CLS life is a journey, not a destination SEP")
# Extract sequence length from the mask
sequence_length = mask.shape[0] #indica il numero di token in una frase (10 in questo esempio)

# Define query, values, and keys with random values and appropriate dimensions
query = torch.randn(sequence_length, sequence_length,embedding_dim)  # (10,10, 768)
values = torch.randn(sequence_length,sequence_length, embedding_dim)  # (10,10, 768)
keys = torch.randn(sequence_length, sequence_length, embedding_dim)   # (10,10, 768)
relative_attention = model(values, keys, query, mask)


print(relative_attention)

# Applica la softmax lungo l'ultimo asse della tua matrice (axis=2)
softmax_attenzione = F.softmax(relative_attention, dim=2)

print(softmax_attenzione)
print(len(relative_attention))








