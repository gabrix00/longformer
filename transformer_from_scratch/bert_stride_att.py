import torch
import torch.nn as nn

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

    def forward(self, values, keys, query, mask, token_index):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)  # (N, value_len, embed_size)
        keys = self.keys(keys)  # (N, key_len, embed_size)
        queries = self.queries(query)  # (N, query_len, embed_size)

        # Select only the embeddings corresponding to the specified token
        queries = queries[:, token_index:token_index+1, :]

        # Calculate q*keys for each training example
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # energy: (N, heads, 1, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention: (N, heads, 1, key_len)

        # Calculate the weighted sum of values using attention
        weighted_values = torch.einsum("nhqk,nkhd->nqhd", [attention, values])
        # weighted_values: (N, heads, 1, head_dim)

        # Merge heads
        weighted_values = weighted_values.reshape(N, 1, self.heads * self.head_dim)
        # weighted_values: (N, 1, embed_size)

        # Apply the final linear layer
        out = self.fc_out(weighted_values)
        # out: (N, 1, embed_size)

        return out.squeeze(dim=1)  # Remove the extra dimension

# Test the modified SelfAttention class
embed_size = 64
heads = 4
seq_len = 10
model = SelfAttention(embed_size, heads)

# Generate some dummy input tensors
values = torch.randn(2, seq_len, embed_size)
keys = torch.randn(2, seq_len, embed_size)
query = torch.randn(2, seq_len, embed_size)
mask = torch.ones(2, seq_len)  # Dummy mask

# Define the index of the token of interest
token_index = 7  # Assuming this is the index of the token "giocare"

# Calculate the relative attention for the specified token
relative_attention = model(values, keys, query, mask, token_index)
print(relative_attention)
