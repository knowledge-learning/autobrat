# coding: utf8

import sys


class EntityAnnotation:
    def __init__(self, line):
        id, mid, text = line.strip().split('\t')
        self.id = id
        typ, spans = mid.split(' ', 1)
        self.type = typ
        self.spans = spans
        self.text = text

    def match(self, other):
        return self.type == other.type and self.spans == other.spans and self.text == other.text

    def __repr__(self):
        return "<Entity(id=%r, type=%r, spans=%r, text=%r)>" % (self.id, self.type, self.spans, self.text)

class AnnFile:
    def __init__(self):
        self.annotations = []

    def load(self, path):
        with open(path) as fp:
            for line in fp:
                self.annotations.append(self._parse(line))

        return self

    def annotations_of(self, type):
        for e in self.annotations:
            if isinstance(e, type):
                yield e

    def _parse(self, line):
        if line.startswith('T'):
            return EntityAnnotation(line)


def mapping(file1: AnnFile, file2: AnnFile):
    map1 = {}
    map2 = {}

    for e1 in file1.annotations_of(EntityAnnotation):
        for e2 in file2.annotations_of(EntityAnnotation):
            if e1.match(e2):
                map1[e1.id] = e2.id
                map2[e2.id] = e1.id

    return map1, map2


def main():
    file1 = AnnFile().load(sys.argv[1])
    file2 = AnnFile().load(sys.argv[2])

    map1, map2 = mapping(file1, file2)
    print(map1)


if __name__ == "__main__":
    main()
