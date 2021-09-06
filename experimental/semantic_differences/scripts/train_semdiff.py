from logging import disable
from pathlib import Path
import spacy
import random
import numpy as np
from pathlib import Path
from typing import List, Optional, Iterable, Tuple, Callable
from pydantic import BaseModel
from spacy.tokens import Doc
from thinc.api import (
    Model,
    Adam,
    SGD,
    CategoricalCrossentropy,
    chain,
    Softmax,
    concatenate,
    Maxout,
    Relu,
    decaying,
    Logistic,
    Linear,
    tuplify,
)
from thinc.layers.add import init as add_init
from thinc.backends import get_array_ops
from thinc.types import Floats2d, cast, Ints1d, Ints2d
from tqdm import tqdm


class AttributeExample(BaseModel):
    term1: str
    term2: str
    attribute: str


class LabeledAttributeExample(BaseModel):
    example: AttributeExample
    label: bool


def dot_products() -> Model[Tuple[Floats2d, Floats2d], Floats2d]:
    return Model(
        "dot_products",
        _dots_forward,
        init=None,
    )


def _dots_forward(
    model: Model[Tuple[Floats2d, Floats2d], Floats2d],
    X1_X2: Tuple[Floats2d, Floats2d],
    is_train: bool,
):
    X1, X2 = X1_X2
    Y = (X1 * X2).sum(axis=1, keepdims=True)

    def backprop_dots(dY: Floats2d) -> Tuple[Floats2d, Floats2d]:
        return (X2 * dY, X1 * dY)

    return Y, backprop_dots


def subtract(
    model1: Model[Floats2d, Floats2d], model2: Model[Floats2d, Floats2d]
) -> Model[Floats2d, Floats2d]:
    # Really we should add a scaling parameter to the addition combinator..
    return Model(
        "subtract",
        _subtract_forward,
        layers=[model1, model2],
        init=add_init,
        dims={"nO": None},
    )


def _subtract_forward(
    model: Model[Tuple[Floats2d, Floats2d], Floats2d],
    X: Floats2d,
    is_train: bool,
) -> Tuple[Model, Callable]:
    Y1, backprop_Y1 = model.layers[0](X, is_train=is_train)
    Y2, backprop_Y2 = model.layers[1](X, is_train=is_train)

    Z = Y1 - Y2

    def _backprop_subtract(dZ: Floats2d) -> Floats2d:
        dX = backprop_Y1(dZ)
        dX += backprop_Y2(-dZ)
        return dX

    return Z, _backprop_subtract


# Bah, fiddling with the typevars here isn't worth the effort.
def getitems(index1: int, index2: int) -> Model[Tuple, Tuple]:
    return Model("getitems", _getitems_forward, attrs={"idx1": index1, "idx2": index2})


def _getitems_forward(model, Xs, is_train: bool):
    idx1 = model.attrs["idx1"]
    idx2 = model.attrs["idx2"]
    Y = (Xs[idx1], Xs[idx2])

    def backprop_getitems(dY: Tuple) -> Tuple:
        dX = [model.ops.alloc2f(*x.shape) for x in Xs]
        dX[idx1] = dY[0]
        dX[idx2] = dY[1]
        return tuple(dX)

    return Y, backprop_getitems


def getitem(index: int) -> Model[Tuple, Floats2d]:
    return Model("getitem", _getitem_forward, attrs={"index": index})


def _getitem_forward(model, Xs, is_train):
    index = model.attrs["index"]
    Y = Xs[index]

    def _getitem_backward(dY: Floats2d) -> Tuple:
        dX = [model.ops.alloc2f(*x.shape) for x in Xs]
        dX[index] = dY
        return tuple(dX)

    return Y, _getitem_backward


def dot_product_model() -> Model[Tuple[Floats2d, Floats2d, Floats2d], Floats2d]:
    """Create a feature vector consisting of three dot products:
    1. x1 * x2
    2. x1 * x3
    3. x2 * x3

    Then learn a two-class softmax layer over these inputs.
    """
    with Model.define_operators({"|": concatenate, ">>": chain, "-": subtract}):
        return (
            (getitems(0, 1) >> dot_products())
            | (getitems(0, 2) >> dot_products())
            | (getitems(1, 2) >> dot_products())
        ) >> Softmax(nO=2)


def difference_model() -> Model[Tuple[Floats2d, Floats2d, Floats2d], Floats2d]:
    """Create a feature vector consisting of three dot products:
    1. x1 * x2
    2. x1 * x3
    3. x2 * x3

    Then learn a two-class softmax layer over these inputs.
    """
    with Model.define_operators(
        {
            "|": concatenate,
            ">>": chain,
            "-": subtract,
            "&": tuplify,
        }
    ):
        return (
            (getitems(0, 1) >> dot_products()) - (getitems(1, 2) >> dot_products())
        ) >> Softmax(nO=2)


def concatenation_model() -> Model[Tuple[Floats2d, Floats2d, Floats2d], Floats2d]:
    """Concatenate word vectors, and pass them into a MLP.

    Then learn a two-class softmax layer over these inputs.
    """
    with Model.define_operators({"|": concatenate, ">>": chain}):
        return (getitem(0) | getitem(1) | getitem(2)) >> Maxout(16) >> Softmax(nO=2)


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
            labeled_example = LabeledAttributeExample(
                example=example, label=is_discriminative
            )
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


def get_word_vectors(nlp, texts: List[str]) -> Floats2d:
    # This way will be a bit faster, especially on GPU, but it's less obvious.
    rows = nlp.vocab.vectors.find(keys=texts)
    word_vecs = nlp.vocab.vectors.data[rows]
    # word_vecs = [normalize_vec(nlp.vocab[text].vector) for text in texts]
    return cast(Floats2d, np.vstack(word_vecs))


def examples_to_arrays(
    examples: Iterable[LabeledAttributeExample],
) -> Tuple[Tuple[Floats2d, Floats2d, Floats2d], Ints2d]:
    """
    Convert a list of DiscriminAtt examples to input and output arrays.

    The rows of the input consist of three dot products, representing the dot product of each
    pair of vectors from (term1, term2, attribute). The output is an Nx1 matrix whose entries
    are 1 or 0 for positive or negative examples.
    """
    # We would recommend using a project file and explicitly loading a proper vectors
    # table as an asset, rather than using the `en_core_web_lg` vectors. But it's okay for now.
    nlp = spacy.load("en_core_web_lg")
    vectors1 = get_word_vectors(nlp, [eg.example.term1 for eg in examples])
    vectors2 = get_word_vectors(nlp, [eg.example.term2 for eg in examples])
    ops = get_array_ops(vectors1)
    assert vectors1.ndim == 2, vectors1.shape
    assert vectors2.ndim == 2, vectors2.shape
    att_vectors = get_word_vectors(nlp, [eg.example.attribute for eg in examples])
    outputs = ops.asarray1i(
        [int(labeled_example.label) for labeled_example in examples], dtype="i"
    )
    assert (
        outputs.shape[0]
        == vectors1.shape[0]
        == vectors2.shape[0]
        == att_vectors.shape[0]
    )

    return (vectors1, vectors2, att_vectors), outputs


def _read_resampled(path: Path, split=0.2):
    examples = list(read_discriminatt_file(path / "train.txt"))
    examples.extend(read_discriminatt_file(path / "validation.txt"))
    random.shuffle(examples)
    dev = examples[: int(len(examples) * split)]
    train = examples[len(dev) :]
    assert len(dev) < len(train)
    assert len(dev) + len(train) == len(examples)
    train_X, train_Y = examples_to_arrays(train)
    val_X, val_Y = examples_to_arrays(dev)
    return train_X, train_Y, val_X, val_Y


def train_model(train_dir: Path, model: Model, optimizer, n_iter, batch_size):
    train_X, train_Y, val_X, val_Y = _read_resampled(train_dir)
    model.initialize(X=train_X[:5], Y=train_Y[:5])
    loss_calc = CategoricalCrossentropy()
    batches = model.ops.multibatch(batch_size, *train_X, train_Y, shuffle=True)
    for n in range(n_iter):
        loss = 0.0
        for X1, X2, X3, Y in tqdm(batches, leave=False):
            Yh, backprop = model.begin_update((X1, X2, X3))
            d_loss = loss_calc.get_grad(Yh, Y).astype("f")
            loss += loss_calc.get_loss(Yh, Y)
            backprop(d_loss)
            model.finish_update(optimizer)
        train_score = evaluate(model, train_X, train_Y, batch_size)
        dev_score = evaluate(model, val_X, val_Y, batch_size)
        print(f"{n}\t{loss:.5f}\t{train_score:.3f}\t{dev_score:.3f}")


def evaluate(model, X1_X2_X3, Y, batch_size):
    correct = 0
    total = 0
    ops = get_array_ops(Y)
    for X1, X2, X3, y in model.ops.multibatch(batch_size, *X1_X2_X3, Y):
        yh = model.predict((X1, X2, X3))
        guesses = yh.argmax(axis=1)
        correct += (guesses == y).sum()
        total += len(yh)
    return correct / total


def evaluate_basic(val_X: Tuple[Floats2d, Floats2d, Floats2d], val_Y):
    """
    Implement a high-performing strategy from SemEval-2018: just compare the dot products
    of the two terms with the attribute.

    This should be an extremely easy function to learn, and on SpaCy's vectors, it gets
    an accuracy of 62.9%. (On vectors retrofit with ConceptNet, it got 73.6%.)
    """
    x1, x2, att = val_X
    dot1 = (x1 * att).sum(axis=-1, keepdims=True)
    dot2 = (x2 * att).sum(axis=-1, keepdims=True)
    yh = (dot1 - dot2) > 0.05
    y = val_Y.reshape((-1, 1))
    correct = ((yh > 0.5) == y).sum()
    total = len(yh)
    assert correct < total, (correct, total)
    return correct / total


def main(
    train_dir: Path,
    out_path: Path,
    *,
    n_iter: int = 40,
    batch_size: int = 32,
    learn_rate: float = 0.0001,
):
    # First, show the result that we can get by already knowing the best simple solution
    val_X, val_Y = examples_to_arrays(
        read_discriminatt_file(train_dir / "validation.txt")
    )
    print(evaluate_basic(val_X, val_Y))

    # Now try to train a machine learning model
    # model = difference_model()
    model = dot_product_model()
    optimizer = Adam(learn_rate=learn_rate, L2=0.0, grad_clip=0.0)
    # optimizer = SGD(learn_rate=0.01, L2=0.0, grad_clip=0.0)
    train_model(train_dir, model, optimizer, n_iter, batch_size)
    model.to_disk(out_path)


if __name__ == "__main__":
    import typer

    typer.run(main)
