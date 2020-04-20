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

from scripts.utils import Collection, Keyphrase, Relation, Sentence
from autobrat.data import (
    load_training_entities,
    load_corpus,
    save_corpus,
    make_sentence,
    load_training_relations,
    _extract_keyphrases_features,
    spacy_model,
)


logger = logging.getLogger("autobrat.classifier")


class Model:
    def __init__(self, corpus):
        self.corpus = corpus
        self.lock = Lock()
        self.unused_sentences = None

        self.entity_classifier = ClassifierEntity()
        self.relation_classifier = ClassifierRelation()

    def warmup(self):
        if self.unused_sentences is not None:
            return

        nlp = spacy_model('es')
        logger.info("Loading unused corpora")
        self.unused_sentences = [nlp(s) for s in load_corpus(self.corpus)]

    def train(self):
        self.warmup()

        self.train_entities()
        self.train_relations()

        if self.lock.locked():
            self.lock.release()

        logger.info("Training finished")

    def train_entities(self):
        # vectorizer
        logger.info("Loading entities training set")
        lines, classes = load_training_entities(self.corpus)

        self.entity_classifier.train_entities(lines, classes)

    def train_relations(self):
        """Entrena el clasificador de relaciones con un par de palabras y 
        la relación correspondiente entre ellas, incluyendo la relación NONE.
        """
        logger.info("Loading relations training set")

        word_pairs, relations = load_training_relations(self.corpus, negative_sampling=0.1)
        self.relation_classifier.train_relations(word_pairs, relations)

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

        return relevant / len(sentence)

    def predict(self, sentences):
        """Predice para cada palabra su etiqueta
        """
        self.warmup()
        collection = self.entity_classifier.predict_entities(sentences)
        collection = self.relation_classifier.predict_relations(collection)

        return collection

    def suggest(self, count=5):
        """Devuelve las k oraciones más relevantes
        """
        self.warmup()

        # procesar corpus de oraciones sin clasificar (vectorizarlas)
        s_list_vector_word = []
        s_words = []

        logger.info("Preparing suggestion pool")

        for s_doc in self.unused_sentences:
            for s_token in s_doc:
                s_words.append(s_token.text)
                s_list_vector_word.append(s_token.vector)

        X_pool_set = np.vstack(s_list_vector_word)

        logger.info("Predicting suggestion score")

        # metric
        weigth = self.classifier.decision_function(X_pool_set)
        relevant_words = {w: f.max() for w, f in zip(s_words, weigth)}

        # relevancia por palabra
        # la seguridad de la clase más problable

        # calcular la relevancia para cada oración
        # teniendo en cuenta la relevancia de cada una de las palabras entre el total de palabras
        # o sea la oración que tiene la mínima confianza en la clase más probable
        relevance = []

        logger.info("Selecting suggestion sentences")

        for s in self.unused_sentences:
            rel = self.relevant_sentence(s, relevant_words)
            relevance.append(rel)

        sorted_relevant_sentences = sorted(
            zip(relevance, self.unused_sentences), key=lambda x: x[0]
        )

        self.unused_sentences = [t[1] for t in sorted_relevant_sentences[count:]]
        save_corpus(self.corpus, self.unused_sentences)

        return sorted_relevant_sentences[:count]


class ClassifierEntity:
    """
    Representa un clasificador de entidades, independiente del corpus.
    Puede ser entrenado con una lista de entidades en formato BILOUV
    y usado para predecir en una lista de oraciones vacías.
    """
    def __init__(self):
        pass

    def predict_entities(self, sentences):
        p_list_vector_word = []
        p_words = []
        p_word_class = {}
        p_list_sentences = []

        p_lines = sentences
        nlp = spacy_model('es')

        for p_sentence in p_lines:
            p_doc = nlp(p_sentence)
            p_list_sentences.append(p_doc)
            for p_token in p_doc:
                p_words.append(p_token.text)
                p_list_vector_word.append(p_token.vector)

        # calcula la predicción para cada palabra
        X_test_set = np.vstack(p_list_vector_word)
        y_test_estimated = self.classifier.predict(X_test_set)

        for i, w in enumerate(p_words):
            p_word_class[w] = y_test_estimated[i]

        # crear la lista de prediciones para cada oración
        result = []
        for s in p_list_sentences:
            cl_s = []
            for t in s:
                cl = p_word_class[t.text]
                cl_s.append(cl)

            # construir una oración de Brat transformando las predicciones
            # en formato BILOV (que es lo que está en `cl_s`) a objetos Keyphrase
            sentence = make_sentence(s, cl_s, self.classes)
            sentence.fix_ids()
            
            result.append(sentence)

        return Collection(sentences=result)


    def train_entities(self, lines, classes):
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

        logger.info(f"Training in {len(X_training_set)} examples")

        # Train classifier
        classifier = SVC(decision_function_shape="ovo")
        classifier.fit(X_training_set, y_training_set)

        self.classes = set(y_training_set)
        self.classifier = classifier


class ClassifierRelation:
    """
    Representa una clasificador de relaciones, independiente del corpus.
    Puede ser entrenado con una lista de pares de palabras -> relación,
    y usado para predecir en una lista de oraciones `Sentence`.
    """
    def __init__(self):
        pass

    def predict_relations(self, collection:Collection):
        nlp = spacy_model('es')

        for sentence in collection.sentences:
            tokens = nlp(sentence.text)
                
            # predecir la relación más probable para cada par de palabras
            for k1 in sentence.keyphrases:
                for k2 in sentence.keyphrases:
                    if k1 == k2:
                        continue

                    # k1 y k2 son Keyphrases, convertir a features
                    vector_pair = _extract_keyphrases_features(tokens, k1, k2)
                    relation_label = self.predict_relation_label(vector_pair)

                    if relation_label:
                        sentence.relations.append(Relation(sentence, k1.id, k2.id, relation_label))
    
        return collection

    def predict_relation_label(self, vector_pair):
        """Predice para cada par de palabras la relación.
        """
        y_test_rel = self.classifier_rel.predict(vector_pair.reshape(1,-1))
        return y_test_rel[0]

    def train_relations(self, word_pairs, relations):
        self.classes_rel = set(relations)

        logger.info(f"Training in {len(word_pairs)} relation pairs")

        # Train classifier
        self.classifier_rel = SVC(decision_function_shape="ovo")
        self.classifier_rel.fit(word_pairs, relations)


if __name__ == "__main__":
    fire.Fire()
