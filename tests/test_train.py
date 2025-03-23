from textgeneration.train import build_dictionary, main_train
from textgeneration.utils.constants import EOL_TOKEN, SOL_TOKEN
from textgeneration.utils.schemas import TrainingInputSchema
import json


def test_n_grams():
    training_args_dict = {
        "trained_model": "./tests/models/",
        "max_n_gram": 3,
        "input_folder": "./tests/corpus/",
    }
    training_args = TrainingInputSchema.model_validate_json(
        json_data=json.dumps(training_args_dict), strict=True
    )
    dictionary = build_dictionary(training_args=training_args)

    # test 1-gram
    assert dictionary[("ALORS",)] == 2
    assert dictionary[("alors",)] == 1
    assert dictionary[("faute",)] == 1
    assert dictionary[("Monsieur",)] == 8
    assert "\n" not in dictionary

    # test 2-grams
    assert dictionary[("son", "père,")] == 6

    # test 3-grams
    assert dictionary[("Monsieur", "B...", "Z...")] == 5

    # test line break
    assert (";", "3") not in dictionary
