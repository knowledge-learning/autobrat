# coding: utf8

import bisect


class Keyphrase:
    def __init__(self, sentence, label, id, spans):
        self.sentence = sentence
        self.label = label
        self.id = id
        self.spans = spans

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

    def clone(self):
        s = Sentence(self.text)
        s.keyphrases = [k.clone(s) for k in self.keyphrases]
        s.relations = [r.clone(s) for r in self.relations]
        return s

    def find_keyphrase(self, id=None, start=None, end=None):
        if id is not None:
            return self._find_keyphrase_by_id(id)
        return self._find_keyphrase_by_spans(start, end)

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

    def _find_keyphrase_by_spans(self, start, end):
        for k in self.keyphrases:
            if k.start == start and k.end == end:
                return k

        return None

    def sort(self):
        self.keyphrases.sort(key=lambda k: tuple([s for s,e in k.spans] + [e for s,e in k.spans]))

    def __len__(self):
        return len(self.text)

    def __repr__(self):
        return "Sentence(text=%r, keyphrases=%r, relations=%r)" % (self.text, self.keyphrases, self.relations)


class Collection:
    def __init__(self, sentences=None):
        self.sentences = sentences or []

    def clone(self):
        return Collection([s.clone() for s in self.sentences])

    def __len__(self):
        return len(self.sentences)

    def dump(self, finput, skip_empty_sentences=True):
        input_file = finput.open('w')
        output_a_file = (finput.parent / ('output_a_' + finput.name[6:])).open('w')
        output_b_file = (finput.parent / ('output_b_' + finput.name[6:])).open('w')

        shift = 0

        for sentence in self.sentences:
            if not sentence.keyphrases and not sentence.relations and skip_empty_sentences:
                continue

            input_file.write("{}\n".format(sentence.text))

            for keyphrase in sentence.keyphrases:
                output_a_file.write("{0}\t{1}\t{2}\t{3}\n".format(
                    keyphrase.id,
                    keyphrase.label,
                    ";".join("{} {}".format(start+shift, end+shift) for start,end in keyphrase.spans),
                    keyphrase.text
                ))

            for relation in sentence.relations:
                output_b_file.write("{0}\t{1}\t{2}\n".format(
                    relation.label,
                    relation.origin,
                    relation.destination
                ))

            shift += len(sentence) + 1


    def load_ann(self, finput):
        ann_file = finput.parent / (finput.name[:-3] + 'ann')
        text = finput.open().read()
        sentences = [s for s in text.split('\n') if s]

        self._parse_ann(sentences, ann_file)

        return len(sentences)

    def _parse_ann(self, sentences, ann_file):
        sentences_length = [len(s) for s in sentences]

        for i in range(1,len(sentences_length)):
            sentences_length[i] += (sentences_length[i-1] + 1)

        sentences_obj = [Sentence(text) for text in sentences]
        labels_by_id = {}
        sentence_by_id = {}

        entities = []
        events = []
        relations = []

        for line in ann_file.open():
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
            # store the annotation in the corresponding sentence
            the_sentence = sentences_obj[i]
            the_sentence.keyphrases.append(Keyphrase(the_sentence, label, lid, spans))
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
                dst_id = event_mapping.get(dst_id, int(dst_id[1:]))

                assert the_sentence == sentence_by_id[dst_id]
                # and store it
                the_sentence.relations.append(Relation(the_sentence, src_id, dst_id, rel_label.lower()))

        for s in sentences_obj:
            s.sort()

        self.sentences.extend(sentences_obj)
