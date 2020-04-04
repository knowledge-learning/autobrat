from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import tqdm
import spacy
import numpy as np

from scripts.utils import Collection
from autobrat.data import load_training_entities, load_corpus


class Model:
    def __init__(self, corpus):
        self.corpus = corpus
        # self.lines = ["Hola a todos como están", "muchas gracias por venir"]
        # self.clases = [["O", "O", "C", "O", "A"], ["O", "O", "O", "A"]]

    def train(self):
        #vectotorizer
        lines, classes = load_training_entities(self.corpus, "Concept")

        list_vector_word = []
        self.words = []
        self.list_sentences = []

        for doc in lines:
            self.list_sentences.append(doc)
            for token in doc:
                self.words.append(token.text)
                list_vector_word.append(token.vector)

        X_training_set = np.vstack(list_vector_word)
        y_training_set = sum(classes, [])

        #Train classifier
        self.classifier = SVC(decision_function_shape='ovo')
        self.classifier.fit(X_training_set, y_training_set)

    def relevant_sentence(self, sentence):
        relevant = 0
        
        for i in sentence:
            relevant += self.relevant_words[i.text]

        return relevant/len(sentence)

    def suggest(self, count=5):
        nlp = spacy.load('es')

        #procesar corpus de oraciones sin clasificar (vectorizarlas)
        s_list_vector_word = []
        s_words = []
        s_list_sentences = []

        select_lines = load_corpus(self.corpus)

        for s_sentence in select_lines:
            s_doc = nlp(s_sentence)
            s_list_sentences.append(s_doc)
            for s_token in s_doc:
                s_words.append(s_token.text)
                s_list_vector_word.append(s_token.vector)
        
        X_pool_set = np.vstack(s_list_vector_word)

        #metric
        weigth = self.classifier.decision_function(X_pool_set)
        self.relevant_words = { w:f.max() for w,f in zip(self.words, weigth) }

        #relevancia por palabra
        #la seguridad de la clase más problable

        #calcular la relevancia para cada oración
        #teniendo en cuenta la relevancia de cada una de las palabras entre el total de palabras
        #o sea la oración que tiene la mínima confianza en la clase más probable
        relevance = []

        for s in self.list_sentences:
            rel = self.relevant_sentence(s)
            relevance.append(rel)

        sorted_relevant_sentences = sorted( zip(relevance, self.list_sentences), key=lambda x: x[0])
        return sorted_relevant_sentences[:5]
        

if __name__ == "__main__":
    model = Model("medline")
    model.train()

    print(model.suggest(5))