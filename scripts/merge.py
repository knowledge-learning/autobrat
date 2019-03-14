# coding: utf8

import sys
import bisect


def offset(id):
    return id[0] + str(int(id[1:]) + 1000)


class EntityAnnotation:
    def __init__(self, line):
        id, mid, text = line.strip().split('\t')
        self.id = id
        typ, spans = mid.split(' ', 1)
        self.type = typ
        self.spans = [tuple(s.split()) for s in spans.split(';')]
        self.text = text

    def __repr__(self):
        return "<Entity(id=%r, type=%r, spans=%r, text=%r)>" % (self.id, self.type, self.spans, self.text)

    def offset_id(self):
        self.id = offset(self.id)

    def as_brat(self):
        spans = ";".join(" ".join(s) for s in self.spans)
        return "%s\t%s %s\t%s" % (self.id, self.type, spans, self.text)

class RelationAnnotation:
    def __init__(self, line):
        id, typ, arg1, arg2 = line.strip().split()
        self.id = id
        self.type = typ
        self.arg1 = arg1.split(':')[1]
        self.arg2 = arg2.split(':')[1]

    def offset_id(self):
        self.arg1 = offset(self.arg1)
        self.arg2 = offset(self.arg2)
        self.id = offset(self.id)

    def __repr__(self):
        return "<Relation(id=%r, type=%r, arg1=%r, arg2=%r)>" % (self.id, self.type, self.arg1, self.arg2)

    def as_brat(self):
        return "%s\t%s Arg1:%s Arg2:%s" % (self.id, self.type, self.arg1, self.arg2)

class SameAsAnnotation(RelationAnnotation):
    def __init__(self, line):
        typ, args = line[1:].strip().split(' ', 1)
        self.id = '*'
        self.type = typ
        self.args = args.split()

    def offset_id(self):
        self.args = [offset(arg) for arg in self.args]

    def __repr__(self):
        return "<Relation(id=%r, type=%r, args=%r)>" % (self.id, self.type, self.args)

    def as_brat(self):
        return "*\t%s %s" % (self.type, " ".join(self.args))

class EventAnnotation:
    def __init__(self, line):
        id, mid = line.strip().split('\t')
        self.id = id
        args = mid.split()
        typ, ref = args[0].split(':')
        args = args[1:]

        self.type = typ
        self.ref = ref
        self.args = { arg.split(':')[0] : arg.split(':')[1] for arg in args }

    def offset_id(self):
        self.ref = offset(self.ref)
        self.id = offset(self.id)

        for k in self.args:
            self.args[k] = offset(self.args[k])

    def __repr__(self):
        return "<Event(id=%r, type=%r, ref=%r, args=%r)>" % (self.id, self.type, self.ref, self.args)

    def as_brat(self):
        spans = " ".join(k + ":" + v for k,v in self.args.items())
        return "%s\t%s:%s %s" % (self.id, self.type, self.ref, spans)

class AttributeAnnotation:
    def __init__(self, line):
        id, typ, ref = line.strip().split()
        self.id = id
        self.type = typ
        self.ref = ref

    def offset_id(self):
        self.ref = offset(self.ref)
        self.id = offset(self.id)

    def __repr__(self):
        return "<Attribute(id=%r, type=%r, ref=%r)>" % (self.id, self.type, self.ref)

    def as_brat(self):
        return "%s\t%s %s" % (self.id, self.type, self.ref)

class AnnFile:
    def __init__(self):
        self.annotations = []

    def load(self, path):
        with open(path) as fp:
            for line in fp:
                ann = self._parse(line)
                if ann:
                    self.annotations.append(ann)

        return self

    def annotations_of(self, type):
        for e in self.annotations:
            if isinstance(e, type):
                yield e

    def offset_spans(self, sentences, first):
        sentences_offset = [-1]

        for s in sentences:
            prev = sentences_offset[-1]
            start = prev + 1
            end = start + len(s)
            sentences_offset.append(end)

        sentences_offset.pop(0)

        for ann in self.annotations_of(EntityAnnotation):
            locations = list(set([bisect.bisect_left(sentences_offset, int(s)) for span in ann.spans for s in span]))

            if len(locations) != 1:
                raise ValueError()

            location = locations.pop()
            offset = sentences_offset[location] + 1

            if first:
                offset = sentences_offset[location - 1] + 1 if location > 0 else 0

            ann.spans = [ (str(int(span[0]) + offset), str(int(span[1]) + offset)) for span in ann.spans ]

    def offset_ids(self):
        for ann in self.annotations:
            ann.offset_id()

    def _parse(self, line):
        if line.startswith('T'):
            return EntityAnnotation(line)

        if line.startswith('R'):
            return RelationAnnotation(line)

        if line.startswith('*'):
            return SameAsAnnotation(line)

        if line.startswith('E'):
            return EventAnnotation(line)

        if line.startswith('A'):
            return AttributeAnnotation(line)

        if line.startswith('#'):
            return None

        raise ValueError("Unknown annotation: %s" % line)


def main():
    file1 = AnnFile().load(sys.argv[1])
    file2 = AnnFile().load(sys.argv[2])
    sents = open(sys.argv[3]).read().split('\n')

    file1.offset_spans(sents, first=True)
    file2.offset_spans(sents, first=False)
    file2.offset_ids()

    for ann in file1.annotations:
        print(ann.as_brat())

    for ann in file2.annotations:
        print(ann.as_brat())


if __name__ == "__main__":
    main()
