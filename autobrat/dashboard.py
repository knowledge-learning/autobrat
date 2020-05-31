import streamlit as st
import altair as alt
import pandas as pd
import json
import numpy as np

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
def load_training_logs():
    data = []

    with open("/data/results_assisted_training.json") as fp:
        for line in fp:
            data.append(json.loads(line))

    return pd.DataFrame(data)


@st.cache
def load_threshold_logs():
    data = []

    with open("/data/results_threshold.json") as fp:
        for line in fp:
            d = {}

            for k,v in json.loads(line).items():
                if isinstance(v, dict):
                    for k2,v2 in v.items():
                        d[f"{k}_{k2}"] = v2
                else:
                    d[k] = v

            data.append(d)

    return pd.DataFrame(data)


@st.cache
def load_collection(corpus):
    return load_training_data(corpus)


MISSING_COST = st.sidebar.number_input("MISSING_COST", 0.0, 10.0, 1.0)
SPURIOUS_COST = st.sidebar.number_input("SPURIOUS_COST", 0.0, 10.0, 2.0)
INCORRECT_COST = st.sidebar.number_input("INCORRECT_COST", 0.0, 10.0, 0.5)
CORRECT_COST = st.sidebar.number_input("CORRECT_COST", 0.0, 10.0, 0.25)
PARTIAL_COST = st.sidebar.number_input("PARTIAL_COST", 0.0, 10.0, 0.25)


def compute_metrics2(
    data,
    missing_cost=MISSING_COST,
    spurious_cost=SPURIOUS_COST,
    incorrect_cost=INCORRECT_COST,
    correct_cost=CORRECT_COST,
    partial_cost=PARTIAL_COST,
    skipA=False,
    skipB=False,
):
    metrics = compute_metrics(data, skipA=skipA, skipB=skipB)

    correct = 0
    partial = 0
    incorrect = 0
    missing = 0
    spurious = 0

    if not skipA:
        correct += len(data["correct_A"])
        incorrect += len(data["incorrect_A"])
        partial += len(data["partial_A"])
        missing += len(data["missing_A"])
        spurious += len(data["spurious_A"])

    if not skipB:
        correct += len(data["correct_B"])
        missing += len(data["missing_B"])
        spurious += len(data["spurious_B"])

    cost = (
        missing_cost * missing
        + spurious_cost * spurious
        + incorrect_cost * incorrect
        + correct_cost * correct
        + partial_cost * partial
    )

    metrics["cost"] = cost
    return metrics


def compute_score(true, predicted):
    dataA = subtaskA(true, predicted)
    dataB = subtaskB(true, predicted, dataA)
    return dict(
        subtaskA=compute_metrics2(dataA, skipB=True),
        subtaskB=compute_metrics2(dataB, skipA=True),
        overall=compute_metrics2(dict(dataA, **dataB)),
    )


def load_all():
    collection = load_collection(corpus)

    training_size = st.number_input(
        "Training size",
        1,
        len(collection.sentences),
        int(len(collection.sentences) / 1.3),
    )

    testing_size = st.number_input(
        "Testing size",
        1,
        len(collection.sentences) - training_size,
        min(len(collection.sentences) - training_size, training_size),
    )

    training_collection = collection[:training_size]
    testing_collection = collection[-testing_size:]

    return collection, training_collection, testing_collection


callback_msg = st.empty()
callback_progress = st.empty()


def callback(msg, current, total):
    callback_msg.markdown(f"{msg}: {current}/{total}")
    callback_progress.progress(current / total)


experiment = st.selectbox(
    "Experiment",
    [
        "Entities",
        "Similarity",
        "Relations",
        "Full training",
        "Assisted comparison",
        "Pre-computed graphs",
    ],
)

if experiment == "Full training":
    collection, training_collection, testing_collection = load_all()

    negative_sampling = st.slider("Negative sampling", 0.0, 1.0, 0.25)
    # max_entity_uncertainty = st.slider("Max entity uncertainty", 0.0, 10.0, 10.0)
    # max_relation_uncertainty = st.slider("Max relation uncertainty", 0.0, 10.0, 10.0)

    model = Model(training_collection, callback, negative_sampling=negative_sampling,)

    #     pool = st.text_area(
    #         "Sentences to score",
    #         """Entre los nutrientes se incluyen las proteínas, carbohidratos, grasas, vitaminas, minerales y agua.
    # El moho está formado por hongos que pueden encontrarse en exteriores o interiores.
    # Puede ser una lumpectomía o una mastectomía.
    # Las estatinas son drogas usadas para bajar el colesterol.
    # Los síndromes mielodisplásicos son poco comunes.""",
    #     ).split("\n")

    if st.button("Train"):
        fp = open("/data/results_threshold_temp.json", "w")

        st.write("### Gold score")
        blank_collection = testing_collection.clone()

        for sentence in blank_collection:
            sentence.keyphrases = []
            sentence.relations = []

        score = compute_score(testing_collection, blank_collection)
        st.write(score)

        model.train()

        for e in np.arange(0, 3, 0.1):
            for r in np.arange(0, 3, 0.1):
                    model.max_entity_uncertainty = e
                    model.max_relation_uncertainty = r

                    st.write(f"### Score with uncertainty for entity={e}; relation={r}")
                    predicted = model.predict(testing_collection.sentences)
                    score = compute_score(testing_collection, predicted)
                    # st.write(score)
                    # st.write(score)
                    score['entity_threshold'] = e
                    score['relation_threshold'] = r
                    fp.write(json.dumps(score) + "\n")
                    fp.flush()

        fp.close()

        # if pool:
        #     for s in pool:
        #         st.write(model.score_sentence(s, return_dict=True))

elif experiment == "Entities":
    collection, training_collection, testing_collection = load_all()

    model = Model(None, callback)

    i = st.slider("Sentence", 0, len(collection) - 1, 0)
    doc, features = model.entity_classifier.feature_sentence(collection[i].text)
    st.write(pd.DataFrame(features))

elif experiment == "Relations":
    collection, training_collection, testing_collection = load_all()

    model = Model(training_collection, callback)

    i = st.slider("Sentence", 0, len(collection) - 1, 0)
    st.code(training_collection.sentences[i].text)
    r = st.selectbox("Relation", training_collection.sentences[i].relations)

    features = model.entity_classifier.relation_features(r)
    st.write(features)

elif experiment == "Similarity":
    collection, training_collection, testing_collection = load_all()

    model = Model(training_collection, callback)
    model.train_similarity()

    correct_0 = 0
    correct_5 = 0

    for i, sentence in enumerate(training_collection.sentences):
        doc, _ = model.entity_classifier.feature_sentence(sentence.text)
        tokens = [token.text for token in doc]
        inferred_vector = model.doc2vec.infer_vector(tokens)
        sims = model.doc2vec.docvecs.most_similar([inferred_vector], topn=5)
        j, _ = sims[0]

        if j == i:
            correct_0 += 1

        if i in [sim[0] for sim in sims]:
            correct_5 += 1

    st.write(
        f"Correct={correct_0}, ({correct_0 / len(training_collection):.2f}), Correct(5)={correct_5}, ({correct_5 / len(training_collection):.2f})"
    )

    i = st.slider("Sentence", 0, len(training_collection) - 1, 0)
    doc, _ = model.entity_classifier.feature_sentence(sentence.text)
    tokens = [token.text for token in doc]
    st.code(tokens)

    inferred_vector = model.doc2vec.infer_vector(tokens)
    st.write(inferred_vector)

    sims = model.doc2vec.docvecs.most_similar([inferred_vector], topn=5)

    for i, v in sims:
        st.code((v, training_collection.sentences[i].text))

elif experiment == "Assisted comparison":
    collection, training_collection, testing_collection = load_all()

    chart = st.altair_chart(
        alt.Chart(pd.DataFrame())
        .mark_line()
        .encode(
            x="batch:Q", y="value:Q", color="type:N", row="metric:N", column="task:N"
        )
        .properties(width=200, height=100,)
    )

    def full_training(
        training_collection: Collection, testing_collection: Collection, batch_size
    ):
        model = Model(training_collection, callback)
        model.train()

        predicted = model.predict(testing_collection.sentences)
        score = compute_score(testing_collection, predicted)

        for batch_end in range(batch_size, len(training_collection), batch_size):
            for task in ["overall", "subtaskA", "subtaskB"]:
                for metric in ["f1", "precision", "recall"]:
                    yield [
                        dict(
                            batch=batch_end,
                            value=score[task][metric],
                            type="full",
                            metric=metric,
                            task=task,
                        ),
                    ]

    def random_training(
        training_collection: Collection, testing_collection: Collection, batch_size
    ):
        for batch_end in range(batch_size, len(training_collection), batch_size):
            batch = training_collection[:batch_end]

            model = Model(batch, callback)
            model.train()
            predicted = model.predict(testing_collection)
            score = compute_score(testing_collection, predicted)
            for task in ["overall", "subtaskA", "subtaskB"]:
                for metric in ["f1", "precision", "recall"]:
                    yield [
                        dict(
                            batch=batch_end,
                            value=score[task][metric],
                            type="random",
                            metric=metric,
                            task=task,
                        ),
                    ]

    def assisted_training(
        training_collection: Collection,
        testing_collection: Collection,
        batch_size,
        suggest_mode,
    ):
        sentences_pool = set([s.text for s in training_collection.sentences])
        training_pool = training_collection[:batch_size]

        model = Model(training_pool, callback, suggest_mode=suggest_mode)
        model.train()

        for batch_end in range(
            batch_size, len(training_collection) - batch_size, batch_size
        ):
            suggestion_pool = list(
                sentences_pool - set(s.text for s in training_pool.sentences)
            )
            suggestions = set(model.suggest(suggestion_pool, batch_size))
            training_pool.sentences.extend(
                s for s in training_collection.sentences if s.text in suggestions
            )

            model = Model(training_pool, callback)
            model.train()

            predicted = model.predict(testing_collection)
            score = compute_score(testing_collection, predicted)
            for task in ["overall", "subtaskA", "subtaskB"]:
                for metric in ["f1", "precision", "recall"]:
                    yield [
                        dict(
                            batch=batch_end,
                            value=score[task][metric],
                            type=f"assisted-{suggest_mode}",
                            metric=metric,
                            task=task,
                        ),
                    ]

    batch_size = st.number_input("Batch size", 1, 100, 10)

    if st.button("Run"):
        all_scores = []
        open("/data/results_assisted_training_temp.json", "w").close()

        for scores in full_training(
            training_collection, testing_collection, batch_size
        ):
            chart.add_rows(scores)
            all_scores.extend(scores)

            with open("/data/results_assisted_training_temp.json", "a") as fp:
                for score in scores:
                    fp.write(json.dumps(score))
                    fp.write("\n")

        for r1, r2, r3, r4 in zip(
            random_training(training_collection, testing_collection, batch_size),
            assisted_training(
                training_collection, testing_collection, batch_size, "full"
            ),
            assisted_training(
                training_collection, testing_collection, batch_size, "entity"
            ),
            assisted_training(
                training_collection, testing_collection, batch_size, "relation"
            ),
        ):
            chart.add_rows(r1 + r2 + r3 + r4)
            all_scores.extend(r1 + r2 + r3 + r4)

            with open("/data/results_assisted_training_temp.json", "a") as fp:
                for score in r1 + r2 + r3 + r4:
                    fp.write(json.dumps(score))
                    fp.write("\n")

        st.write(pd.DataFrame(all_scores))


elif experiment == "Pre-computed graphs":
    data = load_training_logs()

    st.write(data.head(100))

    models = list(data["type"].unique())
    model = st.multiselect("Model", models, models)
    metric = st.selectbox("Metric", data["metric"].unique())
    task = st.selectbox("Task", data["task"].unique())

    batch_min, batch_max = st.slider(
        "Batch range", 0, int(data["batch"].max()), (0, int(data["batch"].max()))
    )

    df = data[
        (data["type"].isin(model))
        & (data["metric"] == metric)
        & (data["task"] == task)
        & (data["batch"] >= batch_min)
        & (data["batch"] <= batch_max)
    ].copy()

    smooth_factor = st.number_input("Smooth factor", 0, 100, 0)
    # TODO: apply smoothing

    st.altair_chart(
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("batch", title="Sentences annotated"),
            y=alt.Y("value", title=metric.title(), scale=alt.Scale(zero=False)),
            color=alt.Color("type", title="Model"),
        ),
        use_container_width=True,
    )

    target = df[df['type'] == 'full']['value'].max()
    steps = [0, 0.8, 0.85, 0.9, 0.95, 1.0]

    df['relative'] = df['value'] / target
    df['relative_clamp'] = df['relative'].apply(lambda x: max(s for s in steps if x >= s))

    df = df[(df['type'] != 'full') & (df['relative_clamp'] >= 0.8)]

    st.altair_chart(
        alt.Chart(df).mark_bar().encode(
            column=alt.Column('relative_clamp:N', title="Relative fitness"),
            x=alt.X('type', title=None),
            y=alt.Y('min(batch)', stack=False),
            color='type',
        )
    )

    st.write(df.groupby(['type', 'relative_clamp']).agg(min_batch=('batch', 'min'), avg_batch=('batch', 'mean')))

    data = load_threshold_logs().copy()
    baseline = data['overall_cost'][0]

    st.write(baseline)

    data['relative_improvement'] =  (baseline - data['overall_cost']) / baseline
    data['relative_improvement_abs'] =  data['relative_improvement'].abs()
    data['improves'] = data['overall_cost'] < baseline
    
    st.write(data)

    st.write("Optimal entity F1: %.3f" % data['subtaskA_f1'].max())
    st.write("Optimal relation F1: %.3f" % data['subtaskB_f1'].max())
    st.write("Optimal cost improvement: %.3f" % data['relative_improvement'].max())

    st.altair_chart(
        alt.Chart(data).mark_circle().encode(
            x=alt.X('entity_threshold:Q', title="Entity threshold"),
            y=alt.Y('relation_threshold:Q', title="Relation threshold"),
            size=alt.Size('relative_improvement_abs', legend=None),
            color=alt.Color('improves', scale=alt.Scale(range=['red', 'green']), legend=None),
        ).properties(
            width=450,
            height=400,
        )
    )
