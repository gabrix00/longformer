import numpy as np
import spacy

import torch
import torch.nn as nn


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
    

### EXPERIMENT #####
'''
 Creat randomly Key, Value and Queries matrix weights
 Set only one head attention
 Set embedding size to 10 only for didactic pourpouse
 Compute attention with the created mask from from_parser2masking
 '''

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
print(softmax_attenzione)
print(len(relative_attention))