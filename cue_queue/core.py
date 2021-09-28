#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Iterable, List, Optional, Type, Union

import fsspec
import numpy as np
from cdp_backend.pipeline.transcript_model import Transcript
from scipy.signal import find_peaks
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from tqdm import tqdm

from .incremental_average import IncrementalAverage

###############################################################################

DEFAULT_TRANSFORMER = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def _load_transformer(
    transformer: Optional[Type[SentenceTransformer]],
) -> SentenceTransformer:
    # Load or pass on transformer
    if transformer is None:
        return SentenceTransformer(DEFAULT_TRANSFORMER)

    return transformer


def _load_transcript(
    transcript: Union[str, Transcript],
) -> Transcript:
    # Read transcript
    if isinstance(transcript, str):
        with fsspec.open(transcript, "r") as open_resource:
            return Transcript.from_json(open_resource.read())

    return transcript


def _load_encoding(
    encoding: Union[str, np.ndarray],
) -> np.ndarray:
    # Read
    if isinstance(encoding, str):
        return np.load(encoding)

    return encoding


def _get_optional_display_iter(
    iterable: Iterable,
    display_progress: bool = True,
    message: Optional[str] = None,
) -> Iterable:
    # Get iterator
    if display_progress:
        return tqdm(iterable, message)

    return iterable


def get_average_delimiter_encoding_for_transcript(
    transcript: Union[str, Transcript],
    transformer: Optional[Type[SentenceTransformer]] = None,
    strict: bool = False,
) -> "np.ndarray":
    """
    Get the average section delimiter encoding for a single transcript.

    Parameters
    ----------
    transcript: Union[str, Transcript]
        The URI for the transcript, or the already loaded transcript,
        to read and process.
    transformer: Optional[Type[SentenceTransformer]]
        An optional transformer to use for generating encodings
        from the delimiter sentences.
        Default: None (use DEFAULT_TRANSFORMER)
    strict: bool
        When True, will raise on any transcript error.
        Default: False (skip problematic sections)

    Returns
    -------
    average_delimiter_sentence_encoding: np.ndarray
        The average delimiter sentence encoding for the provided transcript.
    """
    # TODO:
    # Allow N window around section start to gather

    # Load or use provided transformer
    loaded_transformer = _load_transformer(transformer)

    # Init incremental average
    incremental_average = IncrementalAverage()

    # Read transcript
    loaded_transcript = _load_transcript(transcript)

    # Check section annotations are provided
    if (
        loaded_transcript.annotations is None
        or loaded_transcript.annotations.sections is None
    ):
        raise KeyError("Transcript has no annotations")

    # Get existing sections
    sections = loaded_transcript.annotations.sections

    # Iter section annotations and get section start sentence
    for section in sections:
        try:
            section_start = loaded_transcript.sentences[section.start_sentence_index]

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


def get_average_delimiter_encoding_for_corpus(
    transcripts: Iterable[Union[str, Transcript]],
    transformer: Optional[Type[SentenceTransformer]] = None,
    strict: bool = False,
    display_progress: bool = True,
) -> "np.ndarray":
    """
    Get the average section delimiter encoding for a whole corpus.

    Parameters
    ----------
    transcripts: Iterable[Union[str, Transcript]]
        All transcript URIs, or all loaded transcripts, to read and process.
    transformer: Optional[Type[SentenceTransformer]]
        An optional transformer to use for generating encodings
        from the delimiter sentences.
        Default: None (use DEFAULT_TRANSFORMER)
    strict: bool
        When True, will raise on any transcript error.
        Default: False (skip problematic transcripts)
    display_progress: bool
        When True, will show a progress bar for transcripts processed.
        Default: True (show progress bar)

    Returns
    -------
    average_delimiter_sentence_encoding: np.ndarray
        The average delimiter sentence encoding for the provided corpus.
    """
    # Load or use provided transformer
    loaded_transformer = _load_transformer(transformer)

    # Init incremental average
    incremental_average = IncrementalAverage()

    try:
        # Iterate transcripts
        iterator = _get_optional_display_iter(
            transcripts,
            display_progress=display_progress,
            message="Transcripts processed",
        )
        for transcript in iterator:
            incremental_average.update(
                get_average_delimiter_encoding_for_transcript(
                    transcript=transcript,
                    transformer=loaded_transformer,
                    strict=strict,
                )
            )

    except Exception as e:
        if strict:
            raise e
        else:
            log.error(
                f"Something wrong with transcript: {transcript}, skipping. "
                f"Error: {e}"
            )

    return incremental_average.average


def segment_transcript(
    transcript: Union[str, Transcript],
    delimiter_encoding: Union[str, "np.ndarray"],
    transformer: Optional[Type[SentenceTransformer]] = None,
    overwrite_annotations: bool = True,
    display_progress: bool = True,
) -> Transcript:
    # Load or use provided transformer and encoding
    loaded_transformer = _load_transformer(transformer)
    loaded_delimiter_encoding = _load_encoding(delimiter_encoding)

    # Read transcript
    loaded_transcript = _load_transcript(transcript)

    # Check section annotations do not exist (or overwrite False)
    if (
        loaded_transcript.annotations is not None
        and loaded_transcript.annotations.sections is not None
        and not overwrite_annotations
    ):
        raise KeyError(
            "Transcript has existing section annotations "
            "and `overwrite_annotations` was not set to True."
        )

    # Iter all sentences, get encoding, and store cos sim to average delimiter encoding
    similarities: List[np.ndarray] = []
    iterator = _get_optional_display_iter(
        loaded_transcript.sentences,
        display_progress=display_progress,
        message="Sentences processed",
    )
    for sentence in iterator:
        # Get this sentence encoding and get distance from each section embedding
        sentence_encoding = loaded_transformer.encode(sentence.text)
        similarities.append(
            cos_sim(loaded_delimiter_encoding, sentence_encoding).numpy()[0, 0]
        )

    # Attempt to get segmentation breaks using local maximas
    peaks, _ = find_peaks(np.asarray(similarities))

    return peaks, loaded_transcript