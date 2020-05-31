from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from threading import Thread, Lock

import random
import scipy
import logging
import fire
import uuid
import shutil
import os
import tqdm
import spacy
import numpy as np
import pickle

from sklearn_crfsuite import CRF
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

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
    def __init__(
        self,
        corpus: Collection,
        callback=None,
        language: str = "es",
        negative_sampling: float = 0.25,
        suggest_mode: str = "full",
        max_entity_uncertainty: float = 1e50,
        max_relation_uncertainty: float = 1e50,
    ):
        self.corpus = corpus
        self.lock = Lock()
        self.callback = callback
        self.language = language

        self.entity_classifier = ClassifierEntity(
            callback, negative_sampling=negative_sampling
        )
        self.suggest_mode = suggest_mode
        self.max_entity_uncertainty = max_entity_uncertainty
        self.max_relation_uncertainty = max_relation_uncertainty

    def train(self):
        self.train_similarity()
        self.train_entities()
        self.train_relations()

        if self.lock.locked():
            self.lock.release()

        logger.info("Training finished")

    def train_similarity(self):
        nlp = spacy_model(self.language)
        docs = []

        for i, sentence in enumerate(self.corpus):
            doc = nlp(sentence.text)
            docs.append(TaggedDocument([token.text for token in doc], [i]))

        self.doc2vec = Doc2Vec(docs, min_count=1, epochs=100, vector_size=25)
        self.entity_classifier.doc2vec = self.doc2vec

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

        self.entity_classifier.train_relations(self.corpus)

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

    def predict_entities(self, sentences):
        """Predice para cada palabra su etiqueta
        """
        collection = self.entity_classifier.predict_entities(sentences)

        for sentence in collection:
            sentence.keyphrases = [
                k
                for k in sentence.keyphrases
                if k.uncertainty < self.max_entity_uncertainty
            ]

        return collection

    def predict_relations(self, collection):
        """Predice para cada oración todas las relaciones
        """
        collection = self.entity_classifier.predict_relations(collection)

        for sentence in collection:
            sentence.relations = [
                r
                for r in sentence.relations
                if r.uncertainty < self.max_relation_uncertainty
            ]

        return collection

    def predict(self, sentences):
        return self.predict_relations(self.predict_entities(sentences))

    def score_sentence(self, sentence, return_dict=False):
        if self.suggest_mode == "entity":
            return self.entity_classifier.score_entities(sentence)

        if self.suggest_mode == "relation":
            return self.entity_classifier.score_relations(sentence)

        score_entity = self.entity_classifier.score_entities(sentence)
        score_relation = self.entity_classifier.score_relations(sentence)
        score_similarity = self.entity_classifier.score_similarity(sentence)

        if return_dict:
            return dict(
                score_entity=score_entity,
                score_relations=score_relation,
                score_similarity=score_similarity,
            )

        return 0.5 * (score_entity + score_relation) * score_similarity

    def suggest(self, pool, count=5):
        """Devuelve las k oraciones más relevantes
        """
        scores = {s: self.score_sentence(s) for s in pool}
        scores = sorted(scores, key=scores.get)

        return scores[-count:]


class ClassifierEntity:
    """
    Representa un clasificador de entidades, independiente del corpus.
    Puede ser entrenado con una lista de entidades en formato BILOUV
    y usado para predecir en una lista de oraciones vacías.
    """

    def __init__(self, callback=None, negative_sampling=0.25):
        self.callback = callback
        self.doc2vec = None
        self.negative_sampling = negative_sampling
        self.n_similarity_estimates = 10

    def predict_entities(self, sentences):
        if isinstance(sentences[0], Sentence):
            sentences = [s.text for s in sentences]

        result = []
        nlp = spacy_model("es")

        for i, sentence in enumerate(sentences):
            if self.callback:
                self.callback(
                    msg="Processing sentence", current=i, total=len(sentences)
                )

            doc, xs = self.feature_sentence(sentence)
            sentence = self.predict_single(doc, xs)
            result.append(sentence)

        return Collection(sentences=result)

    def predict_single(self, doc, sequence_of_features):
        labels = self.classifier.predict_single(sequence_of_features)
        sentence = make_sentence(doc, labels, self.classes)
        sentence.fix_ids()

        ys = self.classifier.predict_marginals_single(sequence_of_features)
        entropies = [scipy.stats.entropy(list(yi.values()), base=2) for yi in ys]

        for keyphrase in sentence.keyphrases:
            start = keyphrase.spans[0][0]
            end = keyphrase.spans[-1][1]

            keyphrase_tokens = [
                i
                for i, token in enumerate(doc)
                if token.idx >= start and token.idx + len(token) <= end
            ]
            keyphrase_entropies = [entropies[i] for i in keyphrase_tokens]

            keyphrase.uncertainty = sum(keyphrase_entropies) / len(keyphrase_entropies)

        return sentence

    def score_entities(self, sentence):
        doc, xs = self.feature_sentence(sentence)
        keyphrases = self.predict_single(doc, xs).keyphrases
        entropies = [k.uncertainty for k in keyphrases]
        mean_entropy = sum(entropies) / len(entropies) if entropies else 0

        return mean_entropy

    def score_relations(self, sentence):
        doc, xs = self.feature_sentence(sentence)
        sentence = self.predict_single(doc, xs)
        self.predict_relation_single(doc, sentence)

        entropies = [r.uncertainty for r in sentence.relations]
        mean_entropy = sum(entropies) / len(entropies) if entropies else 0

        return mean_entropy

    def score_similarity(self, sentence):
        tokens = [token.text for token in spacy_model("es")(sentence)]
        inferred_vector = self.doc2vec.infer_vector(tokens)
        sims = [
            v
            for i, v in self.doc2vec.docvecs.most_similar(
                [inferred_vector], topn=self.n_similarity_estimates
            )
        ]
        return np.mean(sims)

    def feature_sentence(self, sentence):
        nlp = spacy_model("es")

        if isinstance(sentence, str):
            doc = nlp(sentence)
        else:
            doc = sentence

        xs = []

        for token in doc:
            xs.append(self.word_features(token))

        return doc, xs

    def word_features(self, word):
        features = dict(
            text=word.text,
            pos=word.pos_,
            dep=word.dep_,
            lemma=word.lemma_,
            entity=word.ent_type_,
            entity_iob=word.ent_iob_,
            kb_id=word.ent_kb_id,
            shape=word.shape_,
            is_alpha=word.is_alpha,
            is_ascii=word.is_ascii,
            is_digit=word.is_digit,
            is_lower=word.is_lower,
            is_upper=word.is_upper,
            is_title=word.is_title,
            is_punct=word.is_punct,
            is_stop=word.is_stop,
            is_left_punct=word.is_left_punct,
            is_right_punct=word.is_right_punct,
            like_url=word.like_url,
            like_num=word.like_num,
            like_email=word.like_email,
        )

        tags = word.tag_

        try:
            _, tags = tags.split("__")
            for tag in tags.split("|"):
                k, v = tag.split("=")
                features[k] = v
        except:
            pass

        return features

    def train_entities(self, sentences, classes):
        logger.info("Preparing training set")

        X_training_set = []
        y_training_set = classes

        for i, sentence in enumerate(sentences):
            doc, xs = self.feature_sentence(sentence)
            X_training_set.append(xs)

            if self.callback:
                self.callback(
                    msg="Processing sentence", current=i, total=len(sentences)
                )

        logger.info(f"Training in {len(X_training_set)} examples")

        # Train classifier
        classifier = CRF()
        classifier.fit(X_training_set, y_training_set)

        self.classes = set(sum(y_training_set, []))
        self.classifier = classifier

    def predict_relation_single(self, doc, sentence):
        # predecir la relación más probable para cada par de palabras
        for k1 in sentence.keyphrases:
            for k2 in sentence.keyphrases:
                if k1 == k2:
                    continue

                # k1 y k2 son Keyphrases, convertir a features
                features = self.relation_features(None, k1, k2, doc)
                relation_label = self.relation_classifier.predict([features])[0]

                if not relation_label:
                    continue

                relation = Relation(sentence, k1.id, k2.id, relation_label)
                probs = self.relation_classifier.predict_proba([features])[0]
                relation.uncertainty = scipy.stats.entropy(list(probs), base=2)
                sentence.relations.append(relation)

    def predict_relations(self, collection: Collection):
        nlp = spacy_model("es")

        for sentence in collection.sentences:
            doc = nlp(sentence.text)
            self.predict_relation_single(doc, sentence)

        return collection

    def relation_features(
        self,
        relation: Relation = None,
        keyphrase_from: Keyphrase = None,
        keyphrase_to: Keyphrase = None,
        doc=None,
    ):
        if relation is not None:
            keyphrase_from = relation.from_phrase
            keyphrase_to = relation.to_phrase

        if doc is None:
            doc = spacy_model("es")(keyphrase_from.sentence.text)

        doc_from = [
            token
            for token in doc
            if token.idx >= keyphrase_from.spans[0][0]
            and token.idx <= keyphrase_from.spans[-1][0]
        ]
        doc_to = [
            token
            for token in doc
            if token.idx >= keyphrase_to.spans[0][0]
            and token.idx <= keyphrase_to.spans[-1][0]
        ]

        from_features = {
            "from_%s" % k: v for k, v in self.word_features(doc_from[0]).items()
        }
        to_features = {"to_%s" % k: v for k, v in self.word_features(doc_to[0]).items()}

        lcp = doc_from[0]

        while not lcp.is_ancestor(doc_to[0]):
            lcp = lcp.head

            if lcp == lcp.head:
                break

        inner_text = [
            token.lemma_ for token in lcp.subtree if token not in doc_to + doc_from
        ]

        d = dict(
            from_features,
            **to_features,
            from_type=keyphrase_from.label,
            to_type=keyphrase_to.label,
        )

        for w in inner_text:
            d[f"inner({w})"] = True

        return d

    def train_relations(self, collection: Collection):
        X_training = []
        y_training = []

        nlp = spacy_model("es")

        for i, sentence in enumerate(collection.sentences):
            doc = nlp(sentence.text)

            for relation in sentence.relations:
                X_training.append(self.relation_features(relation, doc=doc))
                y_training.append(relation.label)

            for k1 in sentence.keyphrases:
                for k2 in sentence.keyphrases:
                    if k1 == k2:
                        continue

                    if (
                        not sentence.find_relations(k1, k2)
                        and random.uniform(0, 1) < self.negative_sampling
                    ):
                        X_training.append(self.relation_features(None, k1, k2, doc))
                        y_training.append("")

            if self.callback:
                self.callback(
                    msg="Processing sentence",
                    current=i,
                    total=len(collection.sentences),
                )

        relation_classifier = make_pipeline(
            DictVectorizer(), LogisticRegression(max_iter=1000)
        )
        relation_classifier.fit(X_training, y_training)

        self.relation_classifier = relation_classifier


if __name__ == "__main__":
    fire.Fire()
