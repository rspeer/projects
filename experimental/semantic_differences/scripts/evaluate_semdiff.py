from pathlib import Path
import json
from train_semdiff import (
    evaluate,
    examples_to_arrays,
    read_discriminatt_file,
    concatenation_model,
    dot_product_model,
)


def main(eval_path: Path, model_path: Path, metrics_path: Path):
    model = dot_product_model().from_disk(model_path)
    val_X, val_Y = examples_to_arrays(read_discriminatt_file(eval_path))
    accuracy = evaluate(model, val_X, val_Y, batch_size=32)
    with metrics_path.open("w") as file_:
        file_.write(json.dumps({"accuracy": accuracy}))
    print("Accuracy: ", accuracy)


if __name__ == "__main__":
    import typer

    typer.run(main)
