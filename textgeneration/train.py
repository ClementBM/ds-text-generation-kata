"""Code for calling the training of the model."""

from sys import argv
import typing

from textgeneration.utils.constants import EOL_TOKEN, SOL_TOKEN
from textgeneration.utils.files import json_to_schema, read_dir, schema_to_json
from textgeneration.utils.preprocessor import Preprocessor
from textgeneration.utils.schemas import (
    DictionarySchema,
    NGramModelSchema,
    TrainingInputSchema,
)


def main_train(file_str_path: str) -> None:
    """
    Call for training an n-gram language model.

    Do not modify its signature.
    You can modify the content.

    :param file_str_path: The path to the JSON that configures the training
    :return: None
    """
    # Reading input data
    training_args = json_to_schema(
        file_str_path=file_str_path, input_schema=TrainingInputSchema
    )

    gram_frequencies = build_dictionary(training_args=training_args)

    dictionary = DictionarySchema(
        max_n_gram=training_args.max_n_gram,
        corpus_files=[
            f
            for f in training_args.input_folder.iterdir()
            if f.is_file() and f.name != ".gitignore"
        ],
        gram_frequencies=gram_frequencies,
        trained_model=training_args.trained_model,
    )

    schema_to_json(
        file_path=training_args.trained_model / "dictionary.json", schema=dictionary
    )

    gram_probabilities = build_model(
        dictionary=dictionary, max_n_gram=training_args.max_n_gram
    )

    model = NGramModelSchema(
        max_n_gram=training_args.max_n_gram,
        corpus_files=[
            f
            for f in training_args.input_folder.iterdir()
            if f.is_file() and f.name != ".gitignore"
        ],
        gram_probabilities=gram_probabilities,
        trained_model=training_args.trained_model,
    )

    schema_to_json(
        file_path=training_args.trained_model / "ngram_model.json", schema=model
    )


def build_model(dictionary: DictionarySchema, max_n_gram: int):
    model: dict = {}

    for n_gram in range(2, max_n_gram + 1):
        model |= build_n_gram_model(dictionary=dictionary, max_n_gram=n_gram)

    return model


def build_n_gram_model(dictionary: DictionarySchema, max_n_gram: int):
    model: dict = {}
    for ngram, freq in dictionary.gram_frequencies.items():
        if len(ngram) == max_n_gram - 1 and freq >= 6:
            model[ngram] = {}

    for ngram, freq in dictionary.gram_frequencies.items():
        if len(ngram) == max_n_gram and freq >= 6:
            model[ngram[:-1]][ngram[-1]] = freq

    n_gram_to_delete = []
    for ngram, probs in model.items():
        total_count = sum(probs.values())
        if total_count == 0:
            n_gram_to_delete.append(ngram)

        for next_gram, freq in probs.items():
            model[ngram][next_gram] = freq / total_count

        model[ngram] = dict(
            sorted(model[ngram].items(), key=lambda item: item[1], reverse=True)
        )

    for ngram in n_gram_to_delete:
        del model[ngram]

    return model


def build_dictionary(training_args: TrainingInputSchema):
    dictionary: dict[tuple, int] = {}

    for training_line in read_dir(dir_path=training_args.input_folder):
        cleaned_line = Preprocessor.clean(text=training_line)

        words = [SOL_TOKEN] + cleaned_line.split() + [EOL_TOKEN]

        update_dictionary(
            dictionary=dictionary,
            grams=words,
            n_gram_count=training_args.max_n_gram,
        )

    return dictionary


def update_dictionary(
    dictionary: dict[tuple, int],
    grams: typing.Union[list[str], tuple],
    n_gram_count: int,
):

    words_slice = slice(None, None if n_gram_count == 1 else 1 - n_gram_count)
    for i, word in enumerate(grams[words_slice]):

        current_grams = tuple(grams[i : i + n_gram_count])

        upsert_dictionary(dictionary, current_grams)

        if n_gram_count == 1:
            continue

        if i == 0:
            update_dictionary(
                dictionary=dictionary,
                grams=current_grams,
                n_gram_count=n_gram_count - 1,
            )
        else:
            sub_grams = current_grams[-n_gram_count + 1 :]
            while len(sub_grams) > 0:
                upsert_dictionary(dictionary, sub_grams)

                if len(sub_grams) == 1:
                    sub_grams = ()
                else:
                    sub_grams = sub_grams[-len(sub_grams) + 1 :]


def upsert_dictionary(dictionary: dict[tuple, int], n_gram: tuple):
    if n_gram in dictionary:
        dictionary[n_gram] += 1
    else:
        dictionary[n_gram] = 1


if __name__ == "__main__":
    """
    poetry run python textgeneration/train.py textgeneration/example_jsons/training.json
    """
    main_train(file_str_path=argv[1])
