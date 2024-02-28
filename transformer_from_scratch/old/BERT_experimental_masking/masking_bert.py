from transformers import BertTokenizer, BertForMaskedLM
import torch
import numpy as np

# Carica il tokenizer di BERT e il modello preaddestrato
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Esempio di testo con un token mascherato
#text = "The capital of France is [MASK]."
text = 'Life is [MASK] journay [MASK] a [MASK]'

# Tokenizza il testo e converte in tensori
tokens = tokenizer(text, return_tensors='pt')

# Calcola la maschera di attenzione
attention_mask = tokens['attention_mask']

# Applica la maschera di attenzione a un token specifico (ad esempio, il token [MASK])
masked_index = torch.where(tokens['input_ids'] == tokenizer.mask_token_id)

print('++++++++++++++++++')
print('tokens are: {}'.format(tokens))
print('attention_mask is : {}'.format(attention_mask))
print('masked_index is : {}'.format(masked_index))
print('++++++++++++++++++')


attention_mask = np.where(tokens['input_ids'] == tokenizer.mask_token_id, 0, attention_mask)

#attention_mask_updated = attention_mask[:, masked_index] = 0  # Imposta a 0 l'attenzione per il token mascherato

print(attention_mask)

'''
# Esegui l'output del modello
outputs = model(input_ids=tokens['input_ids'], attention_mask=attention_mask)

# Recupera i logits per il token mascherato
logits = outputs.logits

# Recupera le probabilità previste dal modello
masked_token_probs = torch.softmax(logits[0, masked_index], dim=-1)

# Recupera l'ID del token previsto con la probabilità massima
predicted_token_ids = torch.argmax(masked_token_probs, dim=-1)

# Converti l'ID del token previsto in forma testuale
predicted_tokens = [tokenizer.decode([predicted_token_id.item()]) for predicted_token_id in predicted_token_ids]

print("Parola prevista:", predicted_tokens)

'''