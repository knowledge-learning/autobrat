# coding: utf8

import bisect
import re
import collections
import warnings

from collections import defaultdict


class Keyphrase:
    def __init__(self, sentence, label, id, spans):
        self.sentence = sentence
        self.label = label
        self.id = id
        self.spans = spans

    def split(self):
        if len(self.spans) > 1:
            raise TypeError("Cannot split a keyphrase with multiple spans")

        start, end = self.spans[0]
        spans = []
        spans.append(start)

        for i, c in enumerate(self.text):
            if c == " ":
                spans.append(start+i)
                spans.append(start+i+1)

        spans.append(end)
        self.spans = [(spans[i],spans[i+1]) for i in range(0, len(spans), 2)]

    def clone(self, sentence):
        return Keyphrase(sentence, self.label, self.id, self.spans)

    @property
    def text(self):
        return " ".join(self.sentence.text[s:e] for (s,e) in self.spans)

    def __repr__(self):
        return "Keyphrase(text=%r, label=%r, id=%r)" % (self.text, self.label, self.id)


class Relation:
    def __init__(self, sentence, origin, destination, label):
        self.sentence = sentence
        self.origin = origin
        self.destination = destination
        self.label = label

    def clone(self, sentence):
        return Relation(sentence, self.origin, self.destination, self.label)

    @property
    def from_phrase(self):
        return self.sentence.find_keyphrase(id=self.origin)

    @property
    def to_phrase(self):
        return self.sentence.find_keyphrase(id=self.destination)

    class _Unk:
        text = 'UNK'

    def __repr__(self):
        from_phrase = (self.from_phrase or Relation._Unk()).text
        to_phrase = (self.to_phrase or Relation._Unk()).text
        return "Relation(from=%r, to=%r, label=%r)" % (from_phrase, to_phrase, self.label)


class Sentence:
    def __init__(self, text):
        self.text = text
        self.keyphrases = []
        self.relations = []

    def clone(self, shallow=False):
        s = Sentence(self.text)
        s.keyphrases = [k if shallow else k.clone(s) for k in self.keyphrases]
        s.relations = [r if shallow else r.clone(s) for r in self.relations]
        return s

    def fix_ids(self, start=1):
        next_id = start

        for k in self.keyphrases:
            for r in self.relations:
                if r.origin == k.id:
                    r.origin = next_id
                if r.destination == k.id:
                    r.destination = next_id

            k.id = next_id
            next_id += 1

        return next_id

    def overlapping_keyphrases(self):
        result = []

        for s1 in self.keyphrases:
            overlaps = set([s1])

            for s2 in self.keyphrases:
                if s2.spans == s1.spans:
                    overlaps.add(s2)

            if len(overlaps) > 1 and overlaps not in result:
                result.append(overlaps)

        return result

    def merge_overlapping_keyphrases(self):
        overlaps = self.overlapping_keyphrases()

        for keyphrases in overlaps:
            keyphrases = list(keyphrases)
            first = keyphrases[0]
            rest = keyphrases[1:]
            rest_ids = [k.id for k in rest]

            for relation in self.relations:
                if relation.origin in rest_ids:
                    print("Changing %r origin from %s to %s" % (relation, relation.origin, first.id))
                    relation.origin = first.id
                if relation.destination in rest_ids:
                    print("Changing %r destination from %s to %s" % (relation, relation.destination, first.id))
                    relation.destination = first.id

            for keyp in rest:
                self.keyphrases.remove(keyp)

    def dup_relations(self):
        dup_relations = collections.defaultdict(lambda: [])

        for r in self.relations:
            dup_relations[(r.label, r.origin, r.destination)].append(r)

        return { k:v for k,v in dup_relations.items() if len(v) > 1}

    def remove_dup_relations(self):
        new_relations = {}

        for r in self.relations:
            new_relations[(r.label, r.origin, r.destination)] = r

        self.relations = list(new_relations.values())

    def find_keyphrase(self, id=None, start=None, end=None, spans=None):
        if id is not None:
            return self._find_keyphrase_by_id(id)
        if spans is None:
            spans = [(start, end)]
        return self._find_keyphrase_by_spans(spans)

    def find_relations(self, orig, dest):
        results = []

        for r in self.relations:
            if r.origin == orig and r.destination == dest:
                results.append(r)

        return results

    def find_relation(self, orig, dest, label):
        for r in self.relations:
            if r.origin == orig and r.destination == dest and label == r.label:
                return r

        return None

    def _find_keyphrase_by_id(self, id):
        for k in self.keyphrases:
            if k.id == id:
                return k

        return None

    def _find_keyphrase_by_spans(self, spans):
        for k in self.keyphrases:
            if k.spans == spans:
                return k

        return None

    def sort(self):
        self.keyphrases.sort(key=lambda k: tuple([s for s,e in k.spans] + [e for s,e in k.spans]))

    def __len__(self):
        return len(self.text)

    def __repr__(self):
        return "Sentence(text=%r, keyphrases=%r, relations=%r)" % (self.text, self.keyphrases, self.relations)

    @staticmethod
    def load(finput):
        return [Sentence(s.strip()) for s in finput.open(encoding='utf8').readlines() if s]


class Collection:
    def __init__(self, sentences=None):
        self.sentences = sentences or []

    def clone(self):
        return Collection([s.clone() for s in self.sentences])

    def __len__(self):
        return len(self.sentences)

    def fix_ids(self):
        next_id = 1

        for s in self.sentences:
            next_id = s.fix_ids(next_id)

    def filter(self, keyphrase=(lambda k: True), relation=(lambda r: True)):
        sentences = []
        for sentence in self.sentences:
            s = Sentence(sentence.text)
            s.keyphrases = [ k.clone(s) for k in sentence.keyphrases if keyphrase(k)]
            s.relations = [ r.clone(s) for r in sentence.relations if keyphrase(r.from_phrase) and keyphrase(r.to_phrase) and relation(r)]
            sentences.append(s)
        return Collection(sentences)
    
    def filter_keyphrase(self, labels):
        return self.filter(keyphrase=lambda k: k.label in labels)

    def filter_relation(self, labels):
        return self.filter(relation=lambda r: r.label in labels)

    def dump(self, finput, skip_empty_sentences=True):
        self.fix_ids()

        input_file = finput.open('w', encoding='utf8')
        output_a_file = (finput.parent / ('output_a_' + finput.name.split("_")[1])).open('w', encoding='utf8')
        output_b_file = (finput.parent / ('output_b_' + finput.name.split("_")[1])).open('w', encoding='utf8')

        shift = 0

        for sentence in self.sentences:
            if not sentence.keyphrases and not sentence.relations and skip_empty_sentences:
                continue

            input_file.write("{}\n".format(sentence.text))

            for keyphrase in sentence.keyphrases:
                output_a_file.write("{0}\t{1}\t{2}\t{3}\n".format(
                    keyphrase.id,
                    ";".join("{} {}".format(start+shift, end+shift) for start,end in keyphrase.spans),
                    keyphrase.label,
                    keyphrase.text
                ))

            for relation in sentence.relations:
                output_b_file.write("{0}\t{1}\t{2}\n".format(
                    relation.label,
                    relation.origin,
                    relation.destination
                ))

            shift += len(sentence) + 1

    def dump_ann(self, finput, skip_empty_sentences=True):
        self.fix_ids()

        def encode(label, num, rel):
            return '{0}{1}:E{2}'.format(
                label,
                num if num else '',
                rel.destination
            )

        rid = 0
        shift = 0
        ann_file = finput.open('w', encoding='utf8')

        for sentence in self.sentences:
            if not sentence.keyphrases and not sentence.relations and skip_empty_sentences:
                continue
            
            relations_from = defaultdict(lambda: defaultdict(list))
            standalone = []
            for relation in sentence.relations:
                if relation.label in ['is-a', 'same-as', 'has-property', \
                                      'part-of', 'causes', 'entails']:
                    standalone.append(relation)
                else:
                    relations_from[relation.origin][relation.label].append(relation)

            for keyphrase in sentence.keyphrases:
                ann_file.write("T{0}\t{1} {2}\t{3}\n".format(
                    keyphrase.id,
                    keyphrase.label,
                    ";".join("{} {}".format(start+shift, end+shift) for start,end in keyphrase.spans),
                    keyphrase.text
                ))

                labels = []
                for label, rels in relations_from[keyphrase.id].items():
                    for i, rel in enumerate(rels):
                        labels.append(encode(label, i, rel))

                ann_file.write("E{0}\t{1}:T{2} {3}\n".format(
                    keyphrase.id,
                    keyphrase.label,
                    keyphrase.id,
                    " ".join(labels),   
                ))
                
            for relation in standalone:
                if relation.label == 'same-as':
                    ann_file.write('*\tsame-as E{0} E{1}\n'.format(
                        relation.origin,
                        relation.destination
                    ))
                else:
                    ann_file.write("R{0}\t{1} Arg1:E{2} Arg2:E{3}\n".format(
                        rid,
                        relation.label,
                        relation.origin,
                        relation.destination
                    ))
                    rid += 1

            shift += len(sentence) + 1

    def load_input(self, finput):
        sentences = [s.strip() for s in finput.open(encoding='utf8').readlines() if s]
        sentences_obj = [Sentence(text) for text in sentences]
        self.sentences.extend(sentences_obj)

    def load_keyphrases(self, finput):
        self.load_input(finput)

        input_a_file = finput.parent / ('output_a_' + finput.name.split("_")[1])

        sentences_length = [len(s.text) for s in self.sentences]
        for i in range(1,len(sentences_length)):
            sentences_length[i] += (sentences_length[i-1] + 1)

        sentence_by_id = {}

        for line in input_a_file.open(encoding='utf8').readlines():
            lid, spans, label, _ = line.strip().split("\t")
            lid = int(lid)

            spans = [s.split() for s in spans.split(";")]
            spans = [(int(start), int(end)) for start, end in spans]

            # find the sentence where this annotation is
            i = bisect.bisect(sentences_length, spans[0][0])
            # correct the annotation spans
            if i > 0:
                spans = [(start - sentences_length[i-1] - 1,
                          end - sentences_length[i-1] - 1)
                          for start,end in spans]
                spans.sort(key=lambda t:t[0])
            # store the annotation in the corresponding sentence
            the_sentence = self.sentences[i]
            keyphrase = Keyphrase(the_sentence, label, lid, spans)
            the_sentence.keyphrases.append(keyphrase)

            if len(keyphrase.spans) == 1:
                keyphrase.split()

            sentence_by_id[lid] = the_sentence

        return sentence_by_id


    def load(self, finput):
        input_b_file = finput.parent / ('output_b_' + finput.name.split("_")[1])

        sentence_by_id = self.load_keyphrases(finput)

        for line in input_b_file.open(encoding='utf8').readlines():
            label, src, dst = line.strip().split("\t")
            src, dst = int(src), int(dst)

            the_sentence = sentence_by_id[src]

            if the_sentence != sentence_by_id[dst]:
                warnings.warn("In file '%s' relation '%s' between %i and %i crosses sentence boundaries and has been ignored." % (finput, label, src, dst))
                continue

            assert sentence_by_id[dst] == the_sentence

            the_sentence.relations.append(Relation(the_sentence, src, dst, label.lower()))

        return self


    def load_ann(self, finput):
        ann_file = finput.parent / (finput.name[:-3] + 'ann')
        text = finput.open(encoding='utf8').read()
        sentences = [s for s in text.split('\n') if s]

        self._parse_ann(sentences, ann_file)

        return self

    def _parse_ann(self, sentences, ann_file):
        sentences_length = [len(s) for s in sentences]

        for i in range(1,len(sentences_length)):
            sentences_length[i] += (sentences_length[i-1] + 1)

        sentences_obj = [Sentence(text) for text in sentences]
        sentence_by_id = {}

        entities = []
        events = []
        relations = []

        for line in ann_file.open(encoding='utf8'):
            if line.startswith('T'):
                entities.append(line)
            elif line.startswith('E'):
                events.append(line)
            elif line.startswith('R') or line.startswith('*'):
                relations.append(line)

        # find all keyphrases
        for entity_line in entities:
            lid, content, text = entity_line.split("\t")
            lid = int(lid[1:])
            label, spans = content.split(" ", 1)
            spans = [s.split() for s in spans.split(";")]
            spans = [(int(start), int(end)) for start, end in spans]

            # find the sentence where this annotation is
            i = bisect.bisect(sentences_length, spans[0][0])
            # correct the annotation spans
            if i > 0:
                spans = [(start - sentences_length[i-1] - 1,
                          end - sentences_length[i-1] - 1)
                          for start,end in spans]
                spans.sort(key=lambda t:t[0])
            # store the annotation in the corresponding sentence
            the_sentence = sentences_obj[i]
            keyphrase = Keyphrase(the_sentence, label, lid, spans)
            the_sentence.keyphrases.append(keyphrase)

            if len(keyphrase.spans) == 1:
                keyphrase.split()

            sentence_by_id[lid] = the_sentence

        event_mapping = {}

        for event_line in events:
            from_id, content = event_line.split("\t")
            to_id = int(content.split()[0].split(":")[1][1:])
            event_mapping[from_id] = to_id

        for event_line in events:
            from_id, content = event_line.split("\t")
            parts = content.split()
            src_id = parts[0].split(":")[1]
            src_id = event_mapping.get(src_id, int(src_id[1:]))
            # find the sentence this relation belongs to
            the_sentence = sentence_by_id[src_id]

            for p in parts[1:]:
                rel_label, dst_id = p.split(":")
                rel_label = result = ''.join([i for i in rel_label if not i.isdigit()])
                dst_id = event_mapping.get(dst_id, int(dst_id[1:]))

                assert the_sentence == sentence_by_id[dst_id]
                # and store it
                the_sentence.relations.append(Relation(the_sentence, src_id, dst_id, rel_label.lower()))

        for relation_line in relations:
            _, content = relation_line.strip().split("\t")
            content = content.split()
            typ, content = content[0], content[1:]

            if typ == 'same-as':
                src_id = content[0]
                src_id = event_mapping.get(src_id, int(src_id[1:]))
                the_sentence = sentence_by_id[src_id]

                for dst_id in content[1:]:
                    dst_id = event_mapping.get(dst_id, int(dst_id[1:]))

                    assert the_sentence == sentence_by_id[dst_id]
                    the_sentence.relations.append(Relation(the_sentence, src_id, dst_id, typ.lower()))

            else:
                src_id = content[0].split(":")[1]
                src_id = event_mapping.get(src_id, int(src_id[1:]))
                the_sentence = sentence_by_id[src_id]

                dst_id = content[1].split(":")[1]
                dst_id = event_mapping.get(dst_id, int(dst_id[1:]))

                assert the_sentence == sentence_by_id[dst_id]
                the_sentence.relations.append(Relation(the_sentence, src_id, dst_id, typ.lower()))

        for s in sentences_obj:
            s.sort()

        self.sentences.extend(sentences_obj)

class DisjointSet:
    def __init__(self, *items):
        self.nodes = { x: DisjointNode(x) for x in items }

    def merge(self, items):
        items = (self.nodes[x] for x in items)
        try:
            head, *others = items
            for other in others:
                head.merge(other)
        except ValueError:
            pass

    @property
    def representatives(self):
        return { n.representative for n in self.nodes.values() }

    @property
    def groups(self):
        return [[n for n in self.nodes.values() if n.representative == r] for r in self.representatives]

    def __len__(self):
        return len(self.representatives)

    def __getitem__(self, item):
        return self.nodes[item]

    def __call__(self, item1, item2):
        return self[item1].representative == self[item2].representative

    def __str__(self):
        return str(self.groups)

    def __repr__(self):
        return str(self)

class DisjointNode:
    def __init__(self, value):
        self.value = value
        self.parent = self

    @property
    def representative(self):
        if self.parent != self:
            self.parent = self.parent.representative
        return self.parent

    def merge(self, other):
        other.representative.parent = self.representative

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self)
