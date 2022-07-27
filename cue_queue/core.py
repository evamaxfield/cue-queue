#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import pickle
import random
from re import split
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple, Union

import fsspec
import numpy as np
from cdp_backend.pipeline.transcript_model import (
    SectionAnnotation,
    Transcript,
    TranscriptAnnotations,
)
from fsspec.core import url_to_fs
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


class ProcessedEncodings(NamedTuple):
    sentence_encodings: np.ndarray
    unified_delimiters_sentence_labels: np.ndarray
    split_delimiters_sentence_labels: np.ndarray
    unified_delimiters_average_encoding: np.ndarray
    start_delimiter_average_encoding: np.ndarray
    stop_delimiter_average_encoding: np.ndarray


###############################################################################


def _load_transformer(
    transformer: Optional[SentenceTransformer],
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


def _load_model(
    model: Union[str, RandomForestClassifier],
) -> RandomForestClassifier:
    # Load or pass
    if isinstance(model, str):
        with fsspec.open(model, "rb") as open_resource:
            return pickle.load(open_resource)

    return model


def _get_optional_display_iter(
    iterable: Iterable,
    display_progress: bool = True,
    message: Optional[str] = None,
) -> Iterable:
    # Get iterator
    if display_progress:
        return tqdm(iterable, message)

    return iterable


# get encodings from transcript
# stack encodings for all delims
# stack encodings for just start delims
# stack encodings for just end delims


def get_single_delim_encodings_for_transcript(
    transcript: Union[str, Transcript],
    transformer: Optional[SentenceTransformer] = None,
    display_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    pass


def get_two_delim_encodings_for_transcript(
    transcript: Union[str, Transcript],
    transformer: Optional[SentenceTransformer] = None,
    display_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    pass


def get_trainable_datas_from_transcript(
    transcript: Union[str, Transcript],
    transformer: Optional[SentenceTransformer] = None,
    display_progress: bool = True,
) -> ProcessedEncodings:
    """
    Get all semantic encodings and labels for a single transcript.

    Parameters
    ----------
    transcript: Union[str, Transcript]
        The URI for the transcript, or the already loaded transcript,
        to read and process.
    transformer: Optional[SentenceTransformer]
        An optional transformer to use for generating semantic encodings.
        Default: None (use DEFAULT_TRANSFORMER)
    display_progress: bool
        When True, will show a progress bar for sentences processed.
        Default: True (show progress bar)

    Returns
    -------
    processed_results: ProcessedEncodings
        All encodings and labels split into many variants for training and application.
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
    start_delimiter_indices = [section.start_sentence_index for section in sections]
    stop_delimiter_indices = [section.stop_sentence_index for section in sections]
    delimiter_indices = start_delimiter_indices + stop_delimiter_indices

    # Iter all sentences and get encodings and label for each
    iterator = _get_optional_display_iter(
        loaded_transcript.sentences,
        display_progress=display_progress,
        message="Sentences processed",
    )
    all_encodings: List[np.ndarray] = []
    unified_delimiter_labels: List[int] = []
    split_delimiter_labels: List[int] = []
    for i, sentence in enumerate(iterator):
        all_encodings.append(
            loaded_transformer.encode(sentence.text, show_progress_bar=False)
        )
        # Append label for unified
        if i in delimiter_indices:
            unified_delimiter_labels.append(1)
        else:
            unified_delimiter_labels.append(0)

        # Append label for split
        if i in start_delimiter_indices:
            split_delimiter_labels.append(1)
        elif i in stop_delimiter_indices:
            split_delimiter_labels.append(2)
        else:
            split_delimiter_labels.append(0)

    # Create all trainable datas and return
    return ProcessedEncodings(
        sentence_encodings=np.asarray(all_encodings),
        unified_delimiters_sentence_labels=np.asarray(unified_delimiter_labels),
        split_delimiters_sentence_labels=np.asarray(split_delimiter_labels),
        unified_delimiters_average_encoding=np.asarray(
            [enc for i, enc in enumerate(all_encodings) if i in delimiter_indices]
        ).mean(axis=0),
        start_delimiter_average_encoding=np.asarray(
            [enc for i, enc in enumerate(all_encodings) if i in start_delimiter_indices]
        ).mean(axis=0),
        stop_delimiter_average_encoding=np.asarray(
            [enc for i, enc in enumerate(all_encodings) if i in stop_delimiter_indices]
        ).mean(axis=0),
    )


def get_trainable_datas_for_corpus(
    transcripts: Iterable[Union[str, Transcript]],
    transformer: Optional[SentenceTransformer] = None,
    strict: bool = False,
    display_progress: bool = True,
) -> ProcessedEncodings:
    """
    Get all semantic encodings and labels for a whole corpus.

    Parameters
    ----------
    transcripts: Iterable[Union[str, Transcript]]
        All transcript URIs, or all loaded transcripts, to read and process.
    transformer: Optional[SentenceTransformer]
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
    processed_corpus: ProcessedEncodings
        All encodings and labels split into many variants for training and application.
    """
    # Load or use provided transformer
    loaded_transformer = _load_transformer(transformer)

    # Iterate transcripts
    iterator = _get_optional_display_iter(
        transcripts,
        display_progress=display_progress,
        message="Transcripts processed",
    )
    processed_encodings: List[ProcessedEncodings] = []
    for transcript in iterator:
        try:
            processed_encodings.append(
                get_trainable_datas_from_transcript(
                    transcript=transcript,
                    transformer=loaded_transformer,
                    display_progress=display_progress,
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

    return ProcessedEncodings(
        sentence_encodings=np.concatenate(
            [pe.sentence_encodings for pe in processed_encodings]
        ),
        unified_delimiters_sentence_labels=np.concatenate(
            [pe.unified_delimiters_sentence_labels for pe in processed_encodings]
        ),
        split_delimiters_sentence_labels=np.concatenate(
            [pe.split_delimiters_sentence_labels for pe in processed_encodings]
        ),
        unified_delimiters_average_encoding=np.stack(
            [pe.unified_delimiters_average_encoding for pe in processed_encodings]
        ).mean(axis=0),
        start_delimiter_average_encoding=np.stack(
            [pe.start_delimiter_average_encoding for pe in processed_encodings]
        ).mean(axis=0),
        stop_delimiter_average_encoding=np.stack(
            [pe.stop_delimiter_average_encoding for pe in processed_encodings]
        ).mean(axis=0),
    )


def train(
    encodings: np.ndarray, labels: np.ndarray, model_kwargs: Dict[str, Any] = {}
) -> Tuple[RandomForestClassifier, np.ndarray]:
    # Init basics
    model = RandomForestClassifier(**model_kwargs)
    model.fit(encodings, labels)
    return model


def eval_model(
    model: RandomForestClassifier,
    encodings: np.ndarray,
    labels: np.ndarray,
    kfold_kwargs: Dict[str, Any] = {
        "n_splits": 10,
        "n_repeats": 3,
        "random_state": 1,
    },
    cross_val_kwargs: Dict[str, Any] = {
        "scoring": "accuracy",
        "n_jobs": -1,
        "error_score": "raise",
    },
) -> np.ndarray:
    cv = RepeatedStratifiedKFold(**kfold_kwargs)
    return cross_val_score(
        model,
        encodings,
        labels,
        cv=cv,
        **cross_val_kwargs,
    )


def segment(
    transcript: Union[str, Transcript],
    model: Union[str, RandomForestClassifier],
    transformer: Optional[SentenceTransformer] = None,
    overwrite_annotations: bool = True,
    display_progress: bool = True,
) -> Transcript:
    # Load or use provided transformer and encoding and model
    loaded_transformer = _load_transformer(transformer)
    loaded_model = _load_model(model)

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

    # Init sections annotations
    if loaded_transcript.annotations is None:
        loaded_transcript.annotations = TranscriptAnnotations()

    loaded_transcript.annotations.sections = []

    # Iter all sentences, get encoding, and store cos sim to average delimiter encoding
    iterator = _get_optional_display_iter(
        loaded_transcript.sentences,
        display_progress=display_progress,
        message="Sentences processed",
    )
    section_start = None
    section_counter = 0
    for sentence in iterator:
        # Get this sentence encoding and get distance from each section embedding
        sentence_encoding = np.expand_dims(
            loaded_transformer.encode(sentence.text, show_progress_bar=False),
            axis=0,
        )
        predicted_label = loaded_model.predict(sentence_encoding)[0]
        if predicted_label == 1:
            if section_start is None:
                section_start = sentence.index
            else:
                loaded_transcript.annotations.sections.append(
                    SectionAnnotation(
                        name=f"Section #{section_counter}",
                        start_sentence_index=section_start,
                        stop_sentence_index=sentence.index,
                        # TODO: Attach version
                        generator="cue-queue",
                    )
                )
                section_start = None
                section_counter += 1

    # Handle last section
    if section_start is not None:
        loaded_transcript.annotations.sections.append(
            SectionAnnotation(
                name=f"Section #{section_counter}",
                start_sentence_index=section_start,
                stop_sentence_index=sentence.index,
                # TODO: Attach version
                generator="cue-queue",
            )
        )

    return loaded_transcript


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


def eval_segmentation(
    true: Union[str, Transcript], pred: Union[str, Transcript]
) -> float:
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


def train_from_corpus(
    corpus_uri: str,
    output_uri: str,
    strict: bool = False,
) -> RandomForestClassifier:
    # Get fs and path specification and corpus list
    log.info("Collecting corpus document URIs.")
    fs, path = url_to_fs(corpus_uri)
    corpus = fs.ls(path)

    # Get encodings and labels for all docs in corpus
    log.info("Beginning encoding of all documents in corpus.")
    encodings, labels = get_encodings_for_corpus(corpus, strict=strict)

    # Train
    log.info("Beginning model training.")
    model = train(encodings=encodings, labels=labels)
    log.info("Completed model training.")

    # Save model
    with fsspec.open(output_uri, "wb") as open_resource:
        pickle.dump(model, open_resource)

    # Eval
    log.info("Beginning model evaluation.")
    score = eval_model(model=model, encodings=encodings, labels=labels)
    log.info("Completed model evaluation.")
    log.info(
        f"Strict model classification accuracy: "
        f"{np.mean(score).round(6)} "
        f"(std: {np.std(score).round(6)})"
    )

    # Sample predict
    log.info("Appling model to sample documents for segmentation accuracy.")
    sample = random.sample(corpus, min([10, len(corpus)]))
    seg_evals_list: List[float] = []
    for doc in sample:
        # Segment
        segmented_transcript = segment(doc, model)

        labels_true = _get_nltk_seg_string_from_transcript(_load_transcript(doc))
        labels_pred = _get_nltk_seg_string_from_transcript(segmented_transcript)
        seg_evals_list.append(
            float(
                boundary_similarity(
                    convert_nltk_to_masses(labels_true),
                    convert_nltk_to_masses(labels_pred),
                )
            )
        )
    seg_evals = np.asarray(seg_evals_list)
    log.info("Completed sample prediction.")
    log.info(
        f"Segmentation accuracy: "
        f"{np.mean(seg_evals).round(6)} "
        f"(std: {np.std(seg_evals).round(6)})"
    )

    return model
