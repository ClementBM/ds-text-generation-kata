import pytest
from textgeneration.generate import generate_text
from textgeneration.train import build_dictionary, main_train
from textgeneration.utils.constants import EOL_TOKEN, SOL_TOKEN
from textgeneration.utils.schemas import InputSchema
import json
import random

random.seed(10)


@pytest.mark.parametrize(
    "input_text,expected",
    [
        pytest.param([], []),
        pytest.param(["Au"], []),
        pytest.param(["Un", "nouveau"], []),
    ],
)
def test_generate(input_text, expected):

    input_args_dict = {
        "trained_model": "./tests/models/",
        "max_n_gram": 3,
        "texts": input_text,
        "output_file": "./tests/models/",
        "use_top_candidate": 5,
    }
    input_args = InputSchema.model_validate_json(
        json_data=json.dumps(input_args_dict), strict=True
    )

    generated_text = generate_text(input_args=input_args)

    # test max length
    assert len(generated_text) - len(input_text) <= 50

    # test special tokens
    assert SOL_TOKEN not in generated_text
    assert EOL_TOKEN not in generated_text

    # test ...
