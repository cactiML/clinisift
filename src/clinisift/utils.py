from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from tqdm import tqdm
import re


def postprocess_wordpiece(ner_results, sentence):
    """
    Need to normalize BERT WordPiece tokenizer and map to words.

    Using following methodology for this:
    - loop through entities
    - if for e1 and e2 e1.end == e2.start, then merge
    into a single entity
    """
    ents = []
    already_parsed = set()

    for i in range(len(ner_results)):
        ent = ner_results[i]
        if ent["word"] not in already_parsed:
            already_parsed.add(ent["word"])
            j = i + 1
            while j < len(ner_results) and ent["end"] == ner_results[j]["start"]:
                ent["end"] = ner_results[j]["end"]
                already_parsed.add(ner_results[j]["word"])
                j += 1
            entity = {
                "word": sentence[ent["start"] : ent["end"]],
                "entity": ent["entity"],
                "start": ent["start"],
                "end": ent["end"],
            }
            ents.append(entity)
    return ents


def iob_parser(iob_str):
    if "-" not in iob_str:
        return (None, iob_str)
    else:
        iob_tag, ent_tag = iob_str.split("-")
        return (iob_tag, ent_tag)


def postprocess_iob(ner_results, sent=None):
    parsed_ents = []
    already_parsed = set()
    for i in range(len(ner_results)):
        if i not in already_parsed:
            already_parsed.add(i)

            ent = ner_results[i]
            iob_tag, ent_tag = iob_parser(ent["entity"])
            merged_ent = ent

            j = i + 1
            if j < len(ner_results):
                next_iob, next_ent = iob_parser(ner_results[j]["entity"])
                while next_iob != "B" and next_ent == ent_tag and j < len(ner_results):
                    already_parsed.add(j)
                    merged_ent["end"] = ner_results[j]["end"]
                    merged_ent["word"] += " " + ner_results[j]["word"]
                    j += 1
                    if j >= len(ner_results):
                        break
                    next_iob, next_ent = iob_parser(ner_results[j]["entity"])

            merged_ent["entity"] = iob_parser(merged_ent["entity"])[1]  # remove iob
            parsed_ents.append(merged_ent)

    # clean gaps in words
    if sent is not None:
        for k in parsed_ents:
            k["word"] = sent[k["start"] : k["end"]]

    return parsed_ents


def ner_sentence(pipeline, sentence, iob_resolve=True, offset=0):
    ner_results = pipeline(sentence)
    ner_results = postprocess_wordpiece(ner_results, sentence)
    if iob_resolve:
        ner_results = postprocess_iob(ner_results, sentence)
    for i in range(len(ner_results)):
        ner_results[i]["start"] += offset
        ner_results[i]["end"] += offset
    return ner_results


def tokenize_sentences(text, sent_tokenizer):
    if isinstance(text, list):
        sents = text
    else:
        sents = sent_tokenizer(text)
    return sents


def identify_sections(expr: str, sents: list):
    sections = []
    curr_heading = "NONE"
    for line in sents:
        h = re.findall(expr, line)
        if len(h) > 0:
            curr_heading = h[0]
        sections.append(curr_heading)
    return sections


# TODO : if guaranteed tokenized sentences, remove `sent_tokenizer` dependency
def ner_doc(
    pipeline,
    sents,
    sent_tokenizer,
    iob_resolve=True,
    include_ents=[],
    exclude_ents=[],
    name="",
):
    ents = []
    for i, sent in enumerate(sents):
        ent_parse = ner_sentence(pipeline, sent, iob_resolve=iob_resolve)
        if len(include_ents) > 0:
            filtered_ents = []
            for e in ent_parse:
                if e["entity"].lower() in include_ents:
                    filtered_ents.append(e)
            ent_parse = filtered_ents

        elif len(exclude_ents) > 0:
            filtered_ents = []
            for e in ent_parse:
                if e["entity"].lower() not in exclude_ents:
                    filtered_ents.append(e)
            ent_parse = filtered_ents

        for e in ent_parse:
            e["model"] = name
            e["sentence"] = i
        ents.append(ent_parse)
    return ents


def ner_visualize(sentences, ner_results, return_html=False):
    import spacy
    from spacy import displacy
    from spacy.tokens import Doc, Span

    assert len(sentences) == len(ner_results)

    ents = []
    offset = 0
    for s, e in zip(sentences, ner_results):
        for ent_ in e:
            ents.append(
                {
                    "start": ent_["start"] + offset,
                    "end": ent_["end"] + offset,
                    "label": ent_["entity"],
                }
            )
        offset += len(s) + 1

    full_text = " ".join(sentences)
    vis = [{"text": full_text, "ents": ents, "title": None}]

    colors = {
        "M": "#208FA8",
        "DO": "#4D5C99",
        "F": "#F0451F",
        "R": "#C77E24",
        "MO": "#7EB0833",
        "PROBLEM": "#208FA8",
        "TREATMENT": "#4D5C99",
        "TEST": "#F0451F",
    }
    options = {"colors": colors}

    if return_html is False:
        displacy.serve(vis, style="ent", manual=True, options=options)
    else:
        html = displacy.render(vis, style="ent", manual=True, options=options)
        return html
