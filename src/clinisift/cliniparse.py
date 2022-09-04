import json
import logging
import torch
from transformers import pipeline, AutoTokenizer


class Parser:
    models = {}
    pipelines = {}
    include_ents = None
    exclude_ents = None
    iob_resolve = None
    sent_tokenizer = None
    sent_per_line = None
    extract_section_headers = None
    section_header_expr = None
    device = None
    defaults = {
        "models": {
            "clinical": "samrawal/bert-base-uncased_clinical-ner",
            "medication": "samrawal/bert-large-uncased_med-ner",
        },
        "section_header_expr": r"^[A-Za-z /]*:",
        "tokenizer_config": {
            "pad_to_max_length": True,
            "model_max_length": 512,
        },
    }

    def __init__(
        self,
        models=None,
        include_ents=[],
        exclude_ents=[],
        iob_resolve=True,
        sent_tokenizer="clinitokenizer",
        sent_per_line=False,
        extract_section_headers=False,
        section_header_expr=None,
        device=None,
    ):
        if models is None:
            default_models = self.defaults["models"]
            self.models.update(default_models)
        elif isinstance(models, list):
            for m in models:
                self.models[m] = m
        elif isinstance(models, dict):
            self.models = models
        else:
            raise TypeError(
                "Parameter 'models' should be None (for defaults), dict, or list."
            )

        self.include_ents = include_ents
        self.exclude_ents = exclude_ents
        if len(include_ents) > 0 and len(exclude_ents) > 0:
            logging.warning(
                "Both include_ents and exclude_ents have values. Overriding exclude_ents, only using include_ents."
            )
            self.exclude_ents = []

        self.iob_resolve = iob_resolve
        try:
            if isinstance(device, str):
                device = int(device.replace("cuda:", ""))
            self.device = device
        except:
            raise TypeError(
                "Error: `device` parameter needs to be integer (GPU number or -1 for CPU)."
            )

        self.sent_tokenizer = self.load_tokenizer(sent_tokenizer)
        self.sent_per_line = sent_per_line
        self.extract_section_headers = extract_section_headers
        if self.extract_section_headers and self.section_header_expr is None:
            self.section_header_expr = self.defaults["section_header_expr"]

        self.configure_pipelines()

    def load_tokenizer(self, sent_tokenizer):
        if sent_tokenizer == "clinitokenizer":
            from clinitokenizer.tokenize import clini_tokenize

            return clini_tokenize
        elif sent_tokenizer == "nltk":
            from nltk.tokenize import sent_tokenize as nltk_tokenize

            return nltk_tokenize
        else:
            raise TypeError(
                "Currently only clinitokenizer and nltk are supported for sent_tokenizer."
            )
            return None

    def configure_pipelines(self):
        if self.device is None:
            gpus = [x for x in range(torch.cuda.device_count())]
            device = -1 if len(gpus) == 0 else gpus[0]
        else:
            device = self.device

        for name, path in self.models.items():
            tokenizer_ = AutoTokenizer.from_pretrained(
                path,
                model_max_length=self.defaults["tokenizer_config"]["model_max_length"],
                pad_to_max_length=self.defaults["tokenizer_config"][
                    "pad_to_max_length"
                ],
            )
            pl = pipeline("ner", model=path, tokenizer=tokenizer_, device=device)
            self.pipelines[name] = pl
