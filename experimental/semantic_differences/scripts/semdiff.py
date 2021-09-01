from logging import disable
import spacy
import numpy as np
from pathlib import Path
from thinc.types import Floats2d, cast
from typing import List, Optional, Iterable, Tuple
from pydantic import BaseModel
from spacy.tokens import Doc

class AttributeExample(BaseModel):
    term1: str
    term2: str
    attribute: str


class LabeledAttributeExample(BaseModel):
    example: AttributeExample
    label: bool


# from torch import nn


# class NeuralDiscriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(300 * 3, 300),
#             nn.ReLU(),
#             nn.Linear(300, 1),
#             nn.Sigmoid()
#         )
    
#     def forward(self, x):
#         logit = self.linear_relu_stack(x)
#         return logit


@spacy.registry.readers("DiscriminAtt.en.v1")
def read_discriminatt_file(filename: Path) -> Iterable[LabeledAttributeExample]:
    examples = []
    for line in open(filename, encoding="utf-8"):
        line = line.strip()
        if line:
            word1, word2, att, target = line.split(",")
            example = AttributeExample(term1=word1, term2=word2, attribute=att)
            is_discriminative = target == "1"
            labeled_example = LabeledAttributeExample(example=example, label=is_discriminative)
            examples.append(labeled_example)
    return examples


def normalize_vec(vec):
    norm = (vec @ vec) ** 0.5
    if norm == 0:
        return vec
    else:
        return vec / norm


def examples_to_arrays(examples: Iterable[LabeledAttributeExample]) -> Tuple[Floats2d, Floats2d]:
    def vector_pipe(texts: List[str]) -> np.ndarray:
        disabled_features = ["tagger", "parser", "ner", "textcat", "lemmatizer"]
        return np.array([normalize_vec(doc.vector) for doc in nlp.pipe(texts, disable=disabled_features)])
        
    # TODO: config
    nlp = spacy.load("en_core_web_lg")
    terms1 = [labeled_example.example.term1 for labeled_example in examples]
    terms2 = [labeled_example.example.term2 for labeled_example in examples]
    attributes = [labeled_example.example.attribute for labeled_example in examples]
    outputs = [[int(labeled_example.label)] for labeled_example in examples]

    vectors1 = vector_pipe(terms1)
    vectors2 = vector_pipe(terms2)
    att_vectors = vector_pipe(attributes)
    input_array = cast(Floats2d, np.hstack([vectors1, vectors2, att_vectors]))
    output_array = cast(Floats2d, np.array(outputs))
    return input_array, output_array

