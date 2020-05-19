import streamlit as st
import altair as alt
import pandas as pd

from pathlib import Path
from autobrat.classifier import Model
from autobrat.data import load_training_data
from scripts.score import subtaskA, subtaskB, compute_metrics
from scripts.utils import Collection


def get_corpora_list():
    return [p for p in Path("/data/").iterdir() if p.is_dir()]


corpora = get_corpora_list()
corpus = st.selectbox("Select corpus", corpora, format_func=lambda p: p.name)


@st.cache
def load_collection(corpus):
    return load_training_data(corpus)


collection = load_collection(corpus)

st.write(f"Total sentences: `{len(collection.sentences)}`")


def compute_score(true, predicted):
    dataA = subtaskA(true, predicted)
    dataB = subtaskB(true, predicted, dataA)
    metrics = compute_metrics(dict(dataA, **dataB))
    return metrics


training_size = st.slider(
    "Training size", 1, len(collection.sentences), 50#int(len(collection.sentences) / 1.3)
)

testing_size = st.slider(
    "Testing size", 1, len(collection.sentences) - training_size, 50#len(collection.sentences) - training_size
)

training_collection = collection[:training_size]
testing_collection = collection[-testing_size:]

if st.selectbox("Experiment", ["Full training", "Assisted comparison"]) == 'Full training':
    model = Model(training_collection)

    pool = st.text_area("Suggestions").split("\n")

    if st.button("Train"):
        model.train()

        predicted = model.predict(testing_collection.sentences)
        score = compute_score(testing_collection, predicted)
        st.write(score)

        suggested = model.suggest(pool)
        st.code(suggested)
else:

    callback_msg = st.empty()
    callback_progress = st.empty()


    def callback(msg, current, total):
        callback_msg.markdown(f"{msg}: {current}/{total}")
        callback_progress.progress(current / total)


    chart = st.altair_chart(
        alt.Chart(
            pd.DataFrame(
                [
                    dict(batch=0, f1=0.0, type="random"),
                    dict(batch=0, f1=0.0, type="assisted"),
                ]
            )
        )
        .mark_line()
        .encode(x="batch", y="f1", color="type"),
        use_container_width=True,
    )


    def full_training(training_collection: Collection, testing_collection: Collection, batch_size):
        model = Model(training_collection, callback)
        model.train()

        predicted = model.predict(testing_collection.sentences)
        score = compute_score(testing_collection, predicted)

        for batch_end in range(batch_size, len(training_collection), batch_size):
            chart.add_rows([dict(batch=batch_end, f1=score["f1"], type="full")])


    def random_training(training_collection: Collection, testing_collection: Collection, batch_size):
        for batch_end in range(batch_size, len(training_collection), batch_size):
            batch = training_collection[:batch_end]

            model = Model(batch, callback)
            model.train()
            predicted = model.predict(testing_collection)
            score = compute_score(testing_collection, predicted)
            chart.add_rows([dict(batch=batch_end, f1=score["f1"], type="random")])


    def assisted_training(training_collection: Collection, testing_collection: Collection, batch_size):
        sentences_pool = set([s.text for s in training_collection.sentences])
        training_pool = training_collection[:batch_size]
        
        model = Model(training_pool, callback)
        model.train()

        for batch_end in range(batch_size, len(training_collection), batch_size):
            suggestion_pool = list(sentences_pool - set(s.text for s in training_pool.sentences))
            suggestions = set(model.suggest(suggestion_pool, batch_size))
            training_pool.sentences.extend(
                s for s in training_collection.sentences if s.text in suggestions
            )

            model = Model(training_pool, callback)
            model.train()
            
            predicted = model.predict(testing_collection)
            score = compute_score(testing_collection, predicted)
            chart.add_rows([dict(batch=batch_end, f1=score["f1"], type="assisted")])


    batch_size = st.number_input("Batch size", 1, 100, 10)

    if st.button("Run full"):
        full_training(training_collection, testing_collection, len(training_collection))

    if st.button("Run"):
        full_training(training_collection, testing_collection, batch_size)
        random_training(training_collection, testing_collection, batch_size)
        assisted_training(training_collection, testing_collection, batch_size)
