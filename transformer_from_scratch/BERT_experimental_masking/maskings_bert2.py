from transformers import BertTokenizer, BertForMaskedLM
import torch

# Carica il tokenizer di BERT e il modello preaddestrato
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Esempio di testo con un token mascherato
text = "[MASK] capital [MASK] France is [MASK]."

# Tokenizza il testo e converte in tensori
tokens = tokenizer(text, return_tensors='pt')

#print('\n\n')
#print('tokens from bert tokenizer are: {}'.format(tokens))

for list in tokens['input_ids'].tolist():
    for t in list:
        print(t)
        print(tokenizer.decode(t))
        print('\n')

#print('decode of tokens are : {}'.formata(

# Calcola la maschera di attenzione
attention_mask = tokens['attention_mask']


# Applica la maschera di attenzione a un token specifico (ad esempio, il token [MASK])
masked_index = torch.where(tokens['input_ids'] == tokenizer.mask_token_id)[1]  # Ottieni solo gli indici di colonna
attention_mask[:, masked_index] = 0  # Imposta a 0 l'attenzione per i token mascherati

# Esegui l'output del modello
outputs = model(input_ids=tokens['input_ids'], attention_mask=attention_mask)


print('++++++++++++++++++')
print('tokens are: {} \n'.format(tokens))
print('attention_mask is : {} \n'.format(attention_mask))
print('masked_index is : {} \n'.format(masked_index))
#print("Esegui l'output del modello : {}".format(outputs))
print('++++++++++++++++++')




# Recupera i logits per il token mascherato
logits = outputs.logits

# Recupera le probabilità previste dal modello
masked_token_probs = torch.softmax(logits[0, masked_index], dim=-1)

# Recupera l'ID del token previsto con la probabilità massima
predicted_token_ids = torch.argmax(masked_token_probs, dim=-1)

# Converti l'ID del token previsto in forma testuale
predicted_tokens = [tokenizer.decode([predicted_token_id.item()]) for predicted_token_id in predicted_token_ids]

print("Parola prevista:", predicted_tokens)

######## plot attention matrix ##########
import matplotlib.pyplot as plt

# Testo e maschera di attenzione
text = "[MASK] capital [MASK] France is [MASK]."
attention_mask = torch.tensor([[1, 0, 1, 0, 1, 1, 0, 1, 1]])

# Indici dei token mascherati
masked_index = torch.tensor([1, 3, 6])

# Tokenizza il testo per ottenere il numero di token
tokens = tokenizer.tokenize(text)

print(tokens) #qui mqnca il token cls eil token sep

#words = tokenizer.decode(tokens)
#print(words)

# Creazione dell'attention matrix
attention_matrix = torch.zeros((len(tokens), len(tokens)))

for i, token_i in enumerate(tokens):
    for j, token_j in enumerate(tokens):
        # Imposta l'attenzione a 0 se uno dei token è mascherato o se ci sono token duplicati
        if i in masked_index or j in masked_index or i == j:
            attention_matrix[i, j] = 0
        else:
            attention_matrix[i, j] = attention_mask[0, min(i, j)]

# Visualizzazione dell'attention matrix
plt.imshow(attention_matrix, cmap='Blues', interpolation='nearest')
plt.title('Attention Matrix')
plt.xlabel('Token Indici')
plt.ylabel('Token Indici')
plt.colorbar()
plt.show()
