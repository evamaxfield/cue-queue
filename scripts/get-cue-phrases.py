#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import re
import string
from pathlib import Path
from typing import List

import pandas as pd
from cdp_backend.pipeline.transcript_model import (
    SectionAnnotation,
    Sentence,
    Transcript,
    TranscriptAnnotations,
)
from nltk import ngrams
from nltk.stem import SnowballStemmer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

###############################################################################
# Globals and model init

ANNOTATED_DATASET = Path("annotated").resolve(strict=True)
STEMMER = SnowballStemmer("english")
TRANSFORMER = SentenceTransformer(
    "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
)

###############################################################################

# For each annotated transcript,
# get three sentences surrounding the start or end sentence for each section,
# create the set of those transcripts
# specifically DO NOT CLEAN but do stem
# take TFIDF to find most common terms specific to start or stop sentences
# I.e. we actually want the least valuable terms for these sections


def get_stemmed_grams(s: Sentence) -> List[str]:
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", s.text)
    grams = [" ".join(gram) for gram in ngrams(text.split(), 1)]
    return [STEMMER.stem(g.lower()) for g in grams]


def get_surrounding_sentences(
    transcript: Transcript,
    center_sentence_index: int,
    look_window: int = 1,
) -> List[Sentence]:
    # Get max sentence index for look right
    transcript_length = len(transcript.sentences)

    # Get look left and right indices
    left = (
        center_sentence_index - look_window
        if center_sentence_index - look_window >= 0
        else 0
    )
    right = (
        center_sentence_index + look_window
        if center_sentence_index + look_window < transcript_length
        else transcript_length - 1
    )

    # Generate semantic encoding that we will average and use to find break points
    _ = TRANSFORMER.encode(" ".join([s.text for s in transcript.sentences[left:right]]))

    # Return section break sentences
    return transcript.sentences[left:right]


def process_transcript(transcript_path: Path) -> pd.DataFrame:
    # Read transcript
    with open(transcript_path, "r") as open_file:
        transcript = Transcript.from_json(open_file.read())  # type: ignore

    # Get sections
    sections = [
        SectionAnnotation.from_dict(s)  # type: ignore
        for s in transcript.annotations[
            TranscriptAnnotations.sections.name  # type: ignore
        ]
    ]

    # Get the stemmed term freqs for each section break sentence sequence
    all_stemmed_grams: List[str] = []
    for section in sections:
        # For each start get stemmed grams and add to list of all for this transcript
        sentences_wrapping_start_of_section = get_surrounding_sentences(
            transcript, section.start_sentence_index
        )
        for sentence in sentences_wrapping_start_of_section:
            all_stemmed_grams += get_stemmed_grams(sentence)

    # Get unique id from filename
    transcript_id = transcript_path.with_suffix("").name
    return pd.DataFrame(
        {
            "transcript_id": [transcript_id for i in range(len(all_stemmed_grams))],
            "stemmed_gram": all_stemmed_grams,
        }
    )


def compute_tfidf_for_corpus(
    corpus: pd.DataFrame, unique_id_col: str, gram_col: str
) -> pd.DataFrame:
    # Get term frequencies
    corpus["tf"] = corpus.groupby([unique_id_col, gram_col])[gram_col].transform(
        "count"
    )

    # Drop duplicates for inverse-document-frequencies
    corpus = corpus.drop_duplicates([unique_id_col, gram_col])

    # Get idf
    N = len(corpus[unique_id_col].unique())
    corpus["idf"] = (
        corpus.groupby(gram_col)[unique_id_col]
        .transform("count")
        .apply(lambda df: math.log(N / df))
    )

    # Store tfidf
    corpus["tfidf"] = corpus.tf * corpus.idf

    return corpus


if __name__ == "__main__":
    # Add all transcript term counts together
    all_transcripts: List[pd.DataFrame] = []

    # Read and process each transcript
    for transcript_path in tqdm(
        list(ANNOTATED_DATASET.glob("*-transcript.json")), "Transcripts processed"
    ):
        try:
            all_transcripts.append(process_transcript(transcript_path))
        except (TypeError, KeyError) as e:
            print(
                f"Something wrong with transcript: {transcript_path}, skipping. "
                f"Error: {e}"
            )

    # Concat and reset index to get all transcripts into a single corpus
    corpus = pd.concat(all_transcripts, ignore_index=True)

    # Compute tfidf
    print("Computing TFIDF scores")
    corpus = compute_tfidf_for_corpus(
        corpus=corpus,
        unique_id_col="transcript_id",
        gram_col="stemmed_gram",
    )

    corpus = corpus.sort_values(by="tfidf", ascending=True)
    corpus.to_csv("cue-phrases-tfidf.csv", index=False)

    print(corpus.head())
