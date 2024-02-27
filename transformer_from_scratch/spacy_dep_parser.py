import numpy as np
import spacy
import matplotlib.pyplot as plt

# Carica il modello linguistico
nlp = spacy.load("en_core_web_sm")

# Definisci la frase di esempio
sentence = nlp("life is a journey, not a destination")

# Creazione della matrice  
num_tokens = len(sentence)
matrix = np.full((num_tokens, num_tokens), -np.inf)

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

            # Imposta a 1 la cella con indice minore o uguale a j+2
            matrix[i][min(len(sentence)-1,j+2)] = 1
            # Imposta a 1 la cella con indice minore o uguale a j+1
            matrix[i][min(len(sentence)-1,j+1)] = 1
            # Imposta a 1 la cella con indice maggiore o uguale a j-2
            matrix[i][max(0,j-2)] = 1
            # Imposta a 1 la cella con indice maggiore o uguale a j-1
            matrix[i][max(0,j-1)] = 1
        else:
            # Se l'indice corrente è uguale all'indice dell'altro token, imposta la cella a 1
            matrix[i][i] = 1
# Stampa della matrice
print(matrix)
'''
# Plot della matrice
list_tokens= [str(token) for token in sentence]
plt.imshow(matrix, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Values')
plt.title('Matrice')
# Aggiungi le etichette per le righe e le colonne
plt.xticks(range(len(list_tokens)), list_tokens, rotation=45)
plt.yticks(range(len(list_tokens)), list_tokens, rotation=45)
plt.show()
'''
