from logging import disable
import spacy
import numpy as np
from pathlib import Path
from typing import List, Optional, Iterable, Tuple
from pydantic import BaseModel
from spacy.tokens import Doc
from thinc.api import Model, Relu, Sigmoid, Adam, CategoricalCrossentropy, chain
from thinc.types import Floats2d, cast
from tqdm import tqdm


class AttributeExample(BaseModel):
    term1: str
    term2: str
    attribute: str


class LabeledAttributeExample(BaseModel):
    example: AttributeExample
    label: bool


# This function should be registered, but I don't know what it should be registered as
def make_classifier(dropout: float=0.2, hidden_width: int=10) -> Model[Floats2d, Floats2d]:
    """
    Make a simple neural classifier that takes in three concatenated word vectors
    and outputs a prediction from 0 to 1.

    The input represents the vectors of two terms and an attribute that may or may not
    distinguish them.

    The output should be near 1 if this is a discriminative attribute for the two terms
    (specifically, an attribute that relates to the first term and not the second),
    and 0 if it is not.
    """
    model: Model[Floats2d, Floats2d] = chain(
        Relu(nO=hidden_width, dropout=dropout),
        Relu(nI=hidden_width, nO=hidden_width, dropout=dropout),
        Relu(nI=hidden_width, nO=1),
        Sigmoid(),
    )
    return model


@spacy.registry.readers("DiscriminAtt.en.v1")  # is this right?
def read_discriminatt_file(filename: Path) -> Iterable[LabeledAttributeExample]:
    """
    Read input in the format provided by the SemEval-2018 DiscriminAtt task.
    An example input line is:

        mushroom,onions,stem,1
    
    indicating that "stem" is an attribute that distinguishes "mushroom" from "onions".
    """
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
    """
    Normalize a vector to a unit vector in the same direction, unless it is the zero vector.
    """
    norm = (vec @ vec) ** 0.5
    if norm == 0:
        return vec
    else:
        return vec / norm


def examples_to_arrays(examples: Iterable[LabeledAttributeExample]) -> Tuple[Floats2d, Floats2d]:
    """
    Convert a list of DiscriminAtt examples to input and output arrays.

    The rows of the input consist of concatenated 300d vectors of the three terms. The
    output is an Nx1 matrix whose entries are 1 or 0 for positive or negative examples.
    """
    def vector_pipe(texts: List[str]) -> np.ndarray:
        disabled_features = ["tagger", "parser", "ner", "textcat", "lemmatizer"]
        return np.array([normalize_vec(doc.vector) for doc in nlp.pipe(texts, disable=disabled_features)])
        
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


def train_model(model, optimizer, n_iter, batch_size):
    train_X, train_Y = examples_to_arrays(read_discriminatt_file("assets/DiscriminAtt/training/train.txt"))
    val_X, val_Y = examples_to_arrays(read_discriminatt_file("assets/DiscriminAtt/training/validation.txt"))
    model.initialize(X=train_X[:5], Y=train_Y[:5])
    loss_calc = CategoricalCrossentropy()

    for n in range(n_iter):
        batches = model.ops.multibatch(batch_size, train_X, train_Y, shuffle=True)
        for X, Y in tqdm(batches, leave=False):
            Yh, backprop = model.begin_update(X)
            d_loss = loss_calc.get_grad(Yh, Y).astype('f')
            loss = loss_calc.get_loss(Yh, Y)
            backprop(d_loss)
            model.finish_update(optimizer)
        
        score = evaluate(model, val_X, val_Y, batch_size)
        print(f"{n}\t{loss:.2f}\t{score:.3f}")
    

def evaluate(model, val_X, val_Y, batch_size):
    correct = 0
    total = 0
    for X, Y in model.ops.multibatch(batch_size, val_X, val_Y):
        Yh = model.predict(X)
        correct += ((Yh > 0.5) == Y).sum()
        total += len(Yh.flatten())
    return correct / total


def run():
    model = make_classifier()
    optimizer = Adam(0.001)
    n_iter = 100
    batch_size = 128
    train_model(model, optimizer, n_iter, batch_size)


if __name__ == '__main__':
    run()
