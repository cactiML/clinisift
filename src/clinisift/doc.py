import os
from clinisift import utils
import logging


class Doc:
    filepath = None
    text = None
    parser = None
    parse_results = {}

    def __init__(self, filepath_or_str, parser, is_file=True):
        if is_file:
            file_ext = os.path.splitext(filepath_or_str)[1]
            if file_ext.lower() != ".txt":
                raise TypeError("Input file must be .txt file.")
            if not os.path.exists(filepath_or_str):
                raise FileNotFoundError(f"File {filepath_or_str} not found.")
            self.filepath = filepath_or_str
            with open(self.filepath, "r") as df:
                self.text = df.readlines()
            if parser.sent_per_line is False:
                self.text = " ".join([s for s in self.text])
        else:
            self.filepath = None
            self.text = filepath_or_str
            if isinstance(self.text, list) and parser.sent_per_line is False:
                logging.warning(
                    "Parser `sent_per_line` is False, but list was passed into Doc. Assuming 1 element in list = 1 sentence. To use sentence tokenizer, pass in text file or string."
                )
        self.parser = parser

    def parse(self):
        self.parse_results = {
            "sentences": None,
            "entities": [],
        }
        self.parse_results["sentences"] = utils.tokenize_sentences(
            self.text, self.parser.sent_tokenizer
        )
        if self.parser.extract_section_headers:
            self.parse_results["sections"] = utils.identify_sections(
                self.parser.section_header_expr, self.parse_results["sentences"]
            )
            assert len(self.parse_results["sections"]) == len(
                self.parse_results["sections"]
            )
        for _ in range(len(self.parse_results["sentences"])):
            self.parse_results["entities"].append([])

        for name, pipeline in self.parser.pipelines.items():
            ner_ents = utils.ner_doc(
                name=name,
                pipeline=pipeline,
                sents=self.parse_results["sentences"],
                sent_tokenizer=self.parser.sent_tokenizer,
                iob_resolve=self.parser.iob_resolve,
                include_ents=self.parser.include_ents,
                exclude_ents=self.parser.exclude_ents,
            )
            assert len(ner_ents) == len(self.parse_results["entities"])

            for i in range(len(ner_ents)):
                self.parse_results["entities"][i] += ner_ents[i]

        self.parse_results["original_text"] = self.text
        self.parse_results["filepath"] = self.filepath

        return self.parse_results

    def get_entities(self, from_model=None):
        entities = [x for y in self.parse_results["entities"] for x in y]
        if from_model is not None:
            entities = [x for x in entities if x["model"] == from_model]
        return entities

    def get_sentence_by_section(self):
        sent_by_section = {}
        # TODO check section parsing done
        if "sections" not in self.parse_results:
            raise TypeError(
                "Section headers not parsed. Please make sure `extract_section_headers=True` when initializing Parser (defaults to False)"
            )
        assert len(self.parse_results["sections"]) == len(
            self.parse_results["sentences"]
        )
        for section, sent in zip(
            self.parse_results["sections"], self.parse_results["sentences"]
        ):
            if section not in sent_by_section:
                sent_by_section[section] = []
            sent_by_section[section].append(sent)
        return sent_by_section

    def keyword_in_ent(self, keywords, match_all=False):
        kw = [keywords] if isinstance(keywords, str) else keywords
        match = []
        for key in kw:
            for ent in self.parse_results["entities"]:
                for ent in sentence:
                    if key.lower() in ent["word"].lower():
                        match.append(True)
                    else:
                        match.append(False)
        return min(match) if match_all else max(match)

    def visualize(self, return_html=False, header=""):
        out = utils.ner_visualize(
            self.parse_results["sentences"],
            self.parse_results["entities"],
            return_html=return_html,
        )
        if return_html:
            return header + "<hr>" + out
