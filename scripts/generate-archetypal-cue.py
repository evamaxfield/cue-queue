#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
from cdp_backend.pipeline.transcript_model import Transcript
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

###############################################################################
# Globals and model init

ANNOTATED_DATASET = Path("annotated").resolve(strict=True)
TRANSFORMER = SentenceTransformer(
    "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
)
ARCHETYPAL_CUE_SENTENCE_ENCODING_PATH = Path("archetypal-cue-sentence.npy")

###############################################################################


def add_to_average(
    current_average: np.ndarray,
    size: int,
    addition: np.ndarray,
) -> np.ndarray:
    return (current_average * size + addition) / (size + 1)


def get_average_cue_sentence_encoding_for_transcript(
    transcript: Transcript,
) -> np.ndarray:
    # TODO:
    # Allow N window around section start to gather

    # Check section annotations are provided
    if transcript.annotations is None or transcript.annotations.sections is None:
        raise KeyError("Transcript has no annotations")

    # Get existing sections
    sections = transcript.annotations.sections

    # Store overall transcript encoding
    transcript_average_cue_sentence_encoding = np.empty(1)
    sections_processed = 0

    # Iter section annotations and get section start sentence
    for section in sections:
        try:
            section_start = transcript.sentences[section.start_sentence_index]

            # Get encoding and add to transcript average
            sentence_embedding = TRANSFORMER.encode(section_start.text)

            # Store or update average
            if sections_processed == 0:
                transcript_average_cue_sentence_encoding = sentence_embedding
            else:
                transcript_average_cue_sentence_encoding = add_to_average(
                    current_average=transcript_average_cue_sentence_encoding,
                    size=sections_processed,
                    addition=sentence_embedding,
                )

            # Update section counter
            sections_processed += 1

        except Exception as e:
            print(
                f"Something went wrong during section processing: "
                f"{section.name}, skipping. "
                f"Error: {e}"
            )

    return transcript_average_cue_sentence_encoding


###############################################################################

if __name__ == "__main__":
    # Store overall average (archetypal) cue sentence encoding
    archetypal_cue_sentence_encoding = None

    # Do not use enumerate, we specifically use a counter here to avoid
    # incorrect incremental averaging in the case a transcript fails
    transcripts_processed = 0

    # Read and process each transcript
    for transcript_path in tqdm(
        list(ANNOTATED_DATASET.glob("*-transcript.json")), "Transcripts processed"
    ):
        try:
            # Read transcript
            with open(transcript_path, "r") as open_file:
                transcript = Transcript.from_json(open_file.read())  # type: ignore

            # Process
            transcript_average_cue_sentence_encoding = (
                get_average_cue_sentence_encoding_for_transcript(
                    transcript=transcript,
                )
            )

            # Store or update average
            if archetypal_cue_sentence_encoding is None:
                archetypal_cue_sentence_encoding = (
                    transcript_average_cue_sentence_encoding
                )
            else:
                archetypal_cue_sentence_encoding = add_to_average(
                    current_average=archetypal_cue_sentence_encoding,
                    size=transcripts_processed,
                    addition=transcript_average_cue_sentence_encoding,
                )

            # Update transcript counter
            transcripts_processed += 1

        except (TypeError, KeyError) as e:
            print(
                f"Something wrong with transcript: {transcript_path}, skipping. "
                f"Error: {e}"
            )

    # Store archetypal cue sentence encoding
    np.save(ARCHETYPAL_CUE_SENTENCE_ENCODING_PATH, archetypal_cue_sentence_encoding)
