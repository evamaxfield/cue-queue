#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

import fsspec
import numpy as np
from cdp_backend.pipeline.transcript_model import Transcript
from segeval import boundary_similarity, convert_nltk_to_masses
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from tqdm import tqdm

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


def _get_optional_display_iter(
    iterable: Iterable,
    display_progress: bool = True,
    message: Optional[str] = None,
) -> Iterable:
    # Get iterator
    if display_progress:
        return tqdm(iterable, message)

    return iterable


def get_encodings_for_transcript(
    transcript: Union[str, Transcript],
    transformer: Optional[Type[SentenceTransformer]] = None,
    display_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get all semantic encodings and labels for a single transcript.

    Parameters
    ----------
    transcript: Union[str, Transcript]
        The URI for the transcript, or the already loaded transcript,
        to read and process.
    transformer: Optional[Type[SentenceTransformer]]
        An optional transformer to use for generating semantic encodings.
        Default: None (use DEFAULT_TRANSFORMER)
    display_progress: bool
        When True, will show a progress bar for sentences processed.
        Default: True (show progress bar)

    Returns
    -------
    encodings: np.ndarray
        All semantic encodings stacked into a single array.
    labels: np.ndarray
        The classification label for each sentence.
    """
    # TODO:
    # Allow N window around section start to gather

    # Load or use provided transformer
    loaded_transformer = _load_transformer(transformer)

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
    delimiter_indices = [section.start_sentence_index for section in sections]

    # Iter all sentences and get encodings and label for each
    iterator = _get_optional_display_iter(
        loaded_transcript.sentences,
        display_progress=display_progress,
        message="Sentences processed",
    )
    encodings: List[np.ndarray] = []
    labels: List[int] = []
    for i, sentence in enumerate(iterator):
        encodings.append(
            loaded_transformer.encode(sentence.text, show_progress_bar=False)
        )
        labels.append(int(i in delimiter_indices))

    return np.asarray(encodings), np.asarray(labels)


def get_encodings_for_corpus(
    transcripts: Iterable[Union[str, Transcript]],
    transformer: Optional[Type[SentenceTransformer]] = None,
    strict: bool = False,
    display_progress: bool = True,
) -> "np.ndarray":
    """
    Get all semantic encodings and labels for a whole corpus.

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
    encodings: np.ndarray
        All semantic encodings stacked into a single array.
    labels: np.ndarray
        The classification label for each sentence in the whole corpus.
    """
    # Load or use provided transformer
    loaded_transformer = _load_transformer(transformer)

    # Iterate transcripts
    iterator = _get_optional_display_iter(
        transcripts,
        display_progress=display_progress,
        message="Transcripts processed",
    )
    all_encodings: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    for transcript in iterator:
        try:
            transcript_encodings, transcript_labels = get_encodings_for_transcript(
                transcript=transcript,
                transformer=loaded_transformer,
                display_progress=display_progress,
            )

            # Append encodings and labels
            all_encodings.append(transcript_encodings)
            all_labels.append(transcript_labels)

        except Exception as e:
            if strict:
                raise e
            else:
                log.error(
                    f"Something wrong with transcript: {transcript}, skipping. "
                    f"Error: {e}"
                )

    return np.concatenate(all_encodings), np.concatenate(all_labels)


def train(
    encodings: np.ndarray, labels: np.ndarray, model_kwargs: Dict[str, Any] = {}
) -> Tuple[Type[RandomForestClassifier], np.ndarray]:
    # Init basics
    model = RandomForestClassifier(**model_kwargs)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(
        model,
        encodings,
        labels,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        error_score="raise",
    )
    return model, n_scores


def segment(
    transcript: Union[str, Transcript],
    model: Type[RandomForestClassifier],
    transformer: Optional[Type[SentenceTransformer]] = None,
    overwrite_annotations: bool = True,
    display_progress: bool = True,
) -> Transcript:
    # Load or use provided transformer and encoding
    loaded_transformer = _load_transformer(transformer)

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
    labels: List[int] = []
    iterator = _get_optional_display_iter(
        loaded_transcript.sentences,
        display_progress=display_progress,
        message="Sentences processed",
    )
    for sentence in iterator:
        # Get this sentence encoding and get distance from each section embedding
        sentence_encoding = loaded_transformer.encode(sentence.text)
        labels.append(model.predict(sentence_encoding))

    return labels


def _get_nltk_seg_string_from_transcript(transcript: Transcript) -> str:
    # Check section annotations are provided
    if transcript.annotations is None or transcript.annotations.sections is None:
        raise KeyError("Transcript has no annotations")

    # Get existing sections
    sections = transcript.annotations.sections
    delimiter_indices = [section.start_sentence_index for section in sections]

    # Get strings
    nltk_seg = ""
    for i, _ in enumerate(transcript.sentences):
        nltk_seg += str(int(i in delimiter_indices))

    return nltk_seg


def eval(true: Union[str, Transcript], pred: Union[str, Transcript]) -> float:
    # Read transcripts
    loaded_true = _load_transcript(true)
    loaded_pred = _load_transcript(pred)

    # Get nltk segmentation strings
    nltk_true = _get_nltk_seg_string_from_transcript(loaded_true)
    nltk_pred = _get_nltk_seg_string_from_transcript(loaded_pred)

    # Return eval
    return float(
        boundary_similarity(
            convert_nltk_to_masses(nltk_true),
            convert_nltk_to_masses(nltk_pred),
        )
    )
