# coding: utf8

from pathlib import Path

from scripts.agreement import load_corpus


def count_complex_entities(path: Path):
    count = 0
    visited = set()
    corpus = load_corpus(path)
    for sentence in corpus.sentences:
        for relation in sentence.relations:
            if (
                relation.from_phrase.id not in visited
                and relation.label in ["subject", "target"]
                and relation.to_phrase.label in ["Action", "Predicate"]
            ):
                visited.add(relation.from_phrase.id)
                count += 1
    return count


def main():
    print(count_complex_entities(Path("./data/v1/medline/phase3-review")))


if __name__ == "__main__":
    main()
