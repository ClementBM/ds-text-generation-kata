"""Code for calling the generating a text."""

from sys import argv
import random

from textgeneration.utils.constants import EOL_TOKEN, SOL_TOKEN
from textgeneration.utils.files import json_to_schema, schema_to_json
from textgeneration.utils.preprocessor import Preprocessor
from textgeneration.utils.schemas import (
    InputSchema,
    NGramModelSchema,
    OutputSchema,
)

"""
Do not print the symbols marking the beginning or ending of a text (see the project's Wiki).
Ensure that the generated text do not start or end with any kind of spaces.
The generation of a text must end when:
    The end of a string has been predicted.
    There is no following word in the model.
    More than 50 words have been generated (do not count those input by the user).
"""


def main_generate(file_str_path: str) -> None:
    """
    Call for generating a text.

    Do not modify its signature.
    You can modify the content.

    :param file_str_path: The path to the JSON that configures the generation
    :return: None
    """
    # Reading input data
    input_args = json_to_schema(file_str_path=file_str_path, input_schema=InputSchema)

    generated_texts = generate_text(input_args=input_args)

    # Printing generated texts
    output_schema = OutputSchema(
        generated_texts=[
            word for word in generated_texts if word not in [EOL_TOKEN, SOL_TOKEN]
        ]
    )
    schema_to_json(
        file_path=input_args.output_file / "output.json", schema=output_schema
    )


def generate_text(input_args: InputSchema):
    model = json_to_schema(
        file_str_path=str(input_args.trained_model / "ngram_model.json"),
        input_schema=NGramModelSchema,
    )

    clean_texts = [SOL_TOKEN]
    for input_text in input_args.texts:
        cleaned_input = Preprocessor.clean(text=input_text)
        if len(cleaned_input) > 0 and cleaned_input not in [SOL_TOKEN, EOL_TOKEN]:
            clean_texts.append(cleaned_input)

    end_of_sequence = False
    input_length = len(clean_texts)

    while not end_of_sequence:
        generated_gram = generate_next_gram(
            model,
            grams=tuple(clean_texts[-input_args.max_n_gram + 1 :]),
            top_k=input_args.use_top_candidate,
        )
        clean_texts.append(generated_gram)

        if len(clean_texts[input_length:]) >= 50 or generated_gram == EOL_TOKEN:
            end_of_sequence = True

    return [word for word in clean_texts if word not in [EOL_TOKEN, SOL_TOKEN]]


def generate_next_gram(model: NGramModelSchema, grams: tuple, top_k: int = 1):
    if grams in model.gram_probabilities:
        top_k_grams = list(model.gram_probabilities[grams].keys())[:top_k]

        return random.choice(top_k_grams)
    else:
        subgrams = grams[-1:] if len(grams) > 1 else ()
        if len(subgrams) == 0:
            return EOL_TOKEN

        return generate_next_gram(model=model, grams=subgrams, top_k=top_k)


if __name__ == "__main__":
    """
    poetry run python textgeneration/generate.py textgeneration/example_jsons/input.json
    """
    main_generate(file_str_path=argv[1])
