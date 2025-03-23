"""JSON Schemas as Pydantic Base Models."""

from pathlib import Path
from typing import Annotated, Tuple
from pydantic import BaseModel, BeforeValidator, Field, PlainSerializer, WithJsonSchema


class _BaseInputSchema(BaseModel):
    """Base Input Schema for our code."""

    trained_model: Annotated[
        Path, Field(description="Indicate where the model is/will be saved")
    ]
    max_n_gram: Annotated[int, Field(description="Maximum n-gram size to use")]


class TrainingInputSchema(_BaseInputSchema):
    """Input Schema for training a model."""

    input_folder: Annotated[
        Path, Field(description="Indicate where the training data is located")
    ]


class InputSchema(_BaseInputSchema):
    """Input Schema for the generation script."""

    texts: Annotated[
        list[str],
        Field(
            description="A list of texts that will be used to generate a longer text"
        ),
    ]
    output_file: Annotated[
        Path, Field(description="Indicates where the generated output should be saved")
    ]
    use_top_candidate: Annotated[
        int,
        Field(
            description=(
                "Indicates from how many top candidates the text can be generated"
            )
        ),
    ] = 1


class OutputSchema(BaseModel):
    """Output Schema for the generation script."""

    generated_texts: Annotated[
        list[str],
        Field(
            description="A list of texts that will be used to generate a longer text"
        ),
    ]


SerializableTuple = Annotated[
    tuple[str, ...],
    BeforeValidator(lambda x: tuple(x.split("/||/")) if isinstance(x, str) else x),
    PlainSerializer(lambda x: "/||/".join(x), return_type=str),
]


class DictionarySchema(_BaseInputSchema):
    corpus_files: Annotated[
        list[Path],
        Field(description="The source text files used to generate the dictionary"),
    ]

    gram_frequencies: Annotated[
        dict[SerializableTuple, int],
        Field(description="N-gram frequencies"),
    ]


class NGramModelSchema(_BaseInputSchema):
    corpus_files: Annotated[
        list[Path],
        Field(description="The source text files used to generate the dictionary"),
    ]

    gram_probabilities: Annotated[
        dict[SerializableTuple, dict[str, float]],
        Field(description="N-gram probabilities"),
    ]
