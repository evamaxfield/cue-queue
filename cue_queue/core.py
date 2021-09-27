#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import TYPE_CHECKING, Iterable, Optional, Type

import fsspec
from cdp_backend.pipeline.transcript_model import Transcript
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .incremental_average import IncrementalAverage

if TYPE_CHECKING:
    import numpy as np

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def _load_transformer(
    transformer: Optional[Type[SentenceTransformer]],
) -> SentenceTransformer:
    # Load or pass on transformer
    if transformer is None:
        return SentenceTransformer(
            "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
        )

    return transformer


def get_average_cue_sentence_encoding_for_transcript(
    transcript_uri: str,
    transformer: Optional[Type[SentenceTransformer]] = None,
    strict: bool = False,
) -> "np.ndarray":
    """
    Get the average cue sentence encoding for a single transcript.

    Parameters
    ----------
    transcript_uri: str
        The URI for the transcript to read and process.
    transformer: Optional[Type[SentenceTransformer]]
        An optional transformer to use for generating encodings from the cue sentences.
        Default: None (use sentence-transformers/paraphrase-xlm-r-multilingual-v1)
    strict: bool
        When True, will raise on any transcript error.
        Default: False (skip problematic sections)

    Returns
    -------
    average_cue_sentence_encoding: np.ndarray
        The average cue sentence encoding for the provided transcript.
    """
    # TODO:
    # Allow N window around section start to gather

    # Load or use provided transformer
    loaded_transformer = _load_transformer(transformer)

    # Init incremental average
    incremental_average = IncrementalAverage()

    # Read transcript
    with fsspec.open(transcript_uri, "r") as open_resource:
        transcript = Transcript.from_json(open_resource.read())

    # Check section annotations are provided
    if transcript.annotations is None or transcript.annotations.sections is None:
        raise KeyError("Transcript has no annotations")

    # Get existing sections
    sections = transcript.annotations.sections

    # Iter section annotations and get section start sentence
    for section in sections:
        try:
            section_start = transcript.sentences[section.start_sentence_index]

            # Get encoding and add to transcript average
            sentence_embedding = loaded_transformer.encode(
                section_start.text, show_progress_bar=False
            )
            incremental_average.update(sentence_embedding)

        except Exception as e:
            if strict:
                raise e
            else:
                log.warning(
                    f"Something went wrong during section processing: "
                    f"{section.name}, skipping. "
                    f"Error: {e}"
                )

    return incremental_average.average


def get_average_cue_sentence_encoding_for_corpus(
    transcript_uris: Iterable[str],
    transformer: Optional[Type[SentenceTransformer]] = None,
    strict: bool = False,
    display_progress: bool = True,
) -> "np.ndarray":
    """
    Get the average cue sentence encoding for a whole corpus.

    Parameters
    ----------
    transcript_uris: str
        All transcript URIs to read and process.
    transformer: Optional[Type[SentenceTransformer]]
        An optional transformer to use for generating encodings from the cue sentences.
        Default: None (use sentence-transformers/paraphrase-xlm-r-multilingual-v1)
    strict: bool
        When True, will raise on any transcript error.
        Default: False (skip problematic transcripts)
    display_progress: bool
        When True, will show a progress bar for transcripts processed.
        Default: True (show progress bar)

    Returns
    -------
    average_cue_sentence_encoding: np.ndarray
        The average cue sentence encoding for the provided corpus.
    """
    # Load or use provided transformer
    loaded_transformer = _load_transformer(transformer)

    # Init incremental average
    incremental_average = IncrementalAverage()

    try:
        # Get iterator
        if display_progress:
            iterator = tqdm(transcript_uris, "Transcripts processed")
        else:
            iterator = transcript_uris

        # Iterate transcripts
        for transcript_uri in iterator:
            incremental_average.update(
                get_average_cue_sentence_encoding_for_transcript(
                    transcript_uri=transcript_uri,
                    transformer=loaded_transformer,
                    strict=strict,
                )
            )

    except Exception as e:
        if strict:
            raise e
        else:
            log.error(
                f"Something wrong with transcript: {transcript_uri}, skipping. "
                f"Error: {e}"
            )

    return incremental_average.average
