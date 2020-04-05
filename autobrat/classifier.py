from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from threading import Thread, Lock

import logging
import fire
import uuid
import shutil
import os
import tqdm
import spacy
import numpy as np
import pickle

from scripts.utils import Collection
from autobrat.data import load_training_entities, load_corpus, save_corpus


logger = logging.getLogger("autobrat.classifier")


class Model:
    def __init__(self, corpus):
        self.corpus = corpus
        self.nlp = None    
        self.lock = Lock()

    def warmup(self):
        if self.nlp is not None:
            return

        logger.info("Creating spacy model")
        self.nlp = spacy.load('es')

        logger.info("Loading unused corpora")
        self.unused_sentences = [self.nlp(s) for s in load_corpus(self.corpus)]

    def train(self):
        self.warmup()

        #vectorizer
        logger.info("Loading training set")
        lines, classes = load_training_entities(self.corpus, "Concept")

        list_vector_word = []
        words = []
        list_sentences = []

        logger.info("Preparing training set")

        for doc in lines:
            list_sentences.append(doc)
            for token in doc:
                words.append(token.text)
                list_vector_word.append(token.vector)

        X_training_set = np.vstack(list_vector_word)
        y_training_set = sum(classes, [])

        logger.info("Training")

        #Train classifier
        classifier = SVC(decision_function_shape='ovo')
        classifier.fit(X_training_set, y_training_set)

        if self.lock.locked():
            self.lock.release()

        logger.info("Training finished")

        self.classifier = classifier

    def train_async(self):
        if self.lock.locked():
            logger.warning("Training in process, skipping this batch.")
            return False

        thread = Thread(target=self.train)
        thread.start()

        return True

    def relevant_sentence(self, sentence, relevant_words):
        relevant = 0
        
        for i in sentence:
            relevant += relevant_words[i.text]

        return relevant/len(sentence)

    def predict(self, sentences):
        self.warmup()

        p_list_vector_word = []
        p_words = []
        p_word_class = {}
        p_list_sentences = []

        p_lines = sentences

        for p_sentence in p_lines:
            p_doc = self.nlp(p_sentence)
            p_list_sentences.append(p_doc)
            for p_token in p_doc:
                p_words.append(p_token.text)
                p_list_vector_word.append(p_token.vector)
        
        #calcula la predicción para cada palabra
        X_test_set = np.vstack(p_list_vector_word)
        y_test_estimated = self.classifier.predict(X_test_set)
        
        for i,w in enumerate(p_words):
            p_word_class[w] = y_test_estimated[i]

        #crear la lista de prediciones para cada oración
        result = []
        for s in p_list_sentences:
            cl_s = []
            for t in s:
                cl = p_word_class[t.text]
                cl_s.append(cl)
            result.append(cl_s)

        return result

    def suggest(self, count=5):
        self.warmup()

        #procesar corpus de oraciones sin clasificar (vectorizarlas)
        s_list_vector_word = []
        s_words = []

        logger.info("Preparing suggestion pool")

        for s_doc in self.unused_sentences:
            for s_token in s_doc:
                s_words.append(s_token.text)
                s_list_vector_word.append(s_token.vector)
        
        X_pool_set = np.vstack(s_list_vector_word)

        logger.info("Predicting suggestion score")

        #metric
        weigth = self.classifier.decision_function(X_pool_set)
        relevant_words = { w:f.max() for w,f in zip(s_words, weigth) }

        #relevancia por palabra
        #la seguridad de la clase más problable

        #calcular la relevancia para cada oración
        #teniendo en cuenta la relevancia de cada una de las palabras entre el total de palabras
        #o sea la oración que tiene la mínima confianza en la clase más probable
        relevance = []

        logger.info("Selecting suggestion sentences")

        for s in self.unused_sentences:
            rel = self.relevant_sentence(s, relevant_words)
            relevance.append(rel)

        sorted_relevant_sentences = sorted(zip(relevance, self.unused_sentences), key=lambda x: x[0])

        self.unused_sentences = [t[1] for t in sorted_relevant_sentences[count:]]
        save_corpus(self.corpus, self.unused_sentences)

        return sorted_relevant_sentences[:count]

# def _train_and_save(model: Model):
#     model.lock.acquire()
#     model.train()

#     name = f"model-{uuid.uuid4()}.pickle"
#     fname = Path("/data") / model.corpus / name

#     with open(fname, "wb") as fp:
#         pickle.Pickler(fp).dump(model)  

#     os.rename(fname, fname.with_name("model.pickle"))

#     model.lock.release()


# def train_and_save(corpus, sync=False):
#     model = Model(corpus)

#     p = Thread(target=_train_and_save, args=(model,))
#     p.start()
    
#     if sync:
#         p.join()

#     return p


# def load_model(corpus):
#     with open(Path("/data") / corpus / "model.pickle", "rb") as fp:
#         return pickle.Unpickler(fp).load()


if __name__ == "__main__":
    fire.Fire()
