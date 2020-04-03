from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import spacy
import numpy as np

#Load datasets

##load training set
#X_training_set = []
#y_training_set = []
#with open(Path(__file__).parent.parent / "data" / "autobrat" / "result") as fp:
#    training_set = fp.read()

##load pool selection set
#X_pool_set = []
#with open(Path(__file__).parent.parent / "data" / "autobrat" / "corpus") as fp:
#    lines = fp.readlines()

lines = ["Hola a todos como están", "muchas gracias por venir"]
clases = [["O", "O", "C", "O", "A"], ["O", "O", "O", "A"]]

#vectotorizer
nlp = spacy.load('es')

list_vector_word = []
words = []
list_sentences = []

for original_sentence in lines:
    doc = nlp(original_sentence)
    list_sentences.append(doc)
    for token in doc:
        words.append(token.text)
        list_vector_word.append(token.vector)

X_training_set = np.vstack(list_vector_word)
y_training_set = sum(clases, [])

#Train classifier
classifier = SVC(decision_function_shape='ovo')
classifier.fit(X_training_set, y_training_set)

#metric
weigth = classifier.decision_function(X_training_set)

#relevancia por palabra
relevant_words = { w:f for w,f in zip(words, weigth) }

#calcular la relevancia para cada oración
#teniendo en cuenta la relevancia de cada una de las palabras entre el total de palabras
def relevant_sentence (sentence):
    relevant = 0
    for i in sentence:
        relevant += relevant_words[i.text]
    return relevant/len(sentence)

relevance = []
for s in list_sentences:
    rel = relevant_sentence(s)
    relevance.append(rel)

print(relevance)

sorted_relevant_sentences = sorted( zip(relevance,list_sentences), key=lambda x: x[0])
print(sorted_relevant_sentences)
