#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cdp_backend.pipeline.transcript_model import Transcript
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from tqdm import tqdm

sns.set_theme(style="darkgrid")

###############################################################################
# Globals and Model Init

ANNOTATED_DATASET = Path("annotated").resolve(strict=True)
PLOTS = Path("plots").resolve()
PLOTS.mkdir(exist_ok=True)
SUMMARIES = Path("summaries").resolve()
SUMMARIES.mkdir(exist_ok=True)
ARCHETYPAL_CUE_SENTENCE_ENCODING_PATH = Path("archetypal-cue-sentence.npy", strict=True)

TRANSFORMER = SentenceTransformer(
    "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
)

###############################################################################


def process_transcript(
    transcript: Transcript,
    archetypal_cue_sentence: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # TODO:
    # Should be provided with num_sections

    # Check transcript is annotated
    if transcript.annotations is None or transcript.annotations.sections is None:
        raise KeyError("Transcript has no annotations")

    # Get N sections
    num_sections = len(transcript.annotations.sections)

    # Process all sentences and store section data as
    # [{"sentence_index": 1, "dist_sim": 5.123}, ... ]
    sentence_data: List[Dict[str, Union[int, str, np.float32]]] = []

    # Iter all sentences, get encoding, compute metrics and get meta
    for sentence in tqdm(transcript.sentences, "Sentences processed"):
        # Get this sentence encoding and get distance from each section embedding
        sentence_encoding = TRANSFORMER.encode(sentence.text)
        sentence_data.append(
            {
                "sentence_index": sentence.index,
                "sentence_text": sentence.text,
                "dist_sim": np.linalg.norm(archetypal_cue_sentence - sentence_encoding),
                "cos_sim": cos_sim(archetypal_cue_sentence, sentence_encoding).numpy()[
                    0, 0
                ],
            }
        )

    # Convert to dataframe
    sentence_distances = pd.DataFrame(sentence_data)

    # Get closest N sentences and return
    selected_sentences = sentence_distances.sort_values(by="cos_sim", ascending=False)[
        :num_sections
    ]
    return sentence_distances, selected_sentences


###############################################################################

if __name__ == "__main__":
    # Load archetype
    archetypal_cue_sentence = np.load(ARCHETYPAL_CUE_SENTENCE_ENCODING_PATH)

    # Read and process each transcript
    for transcript_path in tqdm(
        list(ANNOTATED_DATASET.glob("*-transcript.json")), "Transcripts processed"
    ):
        try:
            # Read transcript
            with open(transcript_path, "r") as open_file:
                transcript = Transcript.from_json(open_file.read())  # type: ignore

            # Process
            all_transcript_sentence_distances, selected_breaks = process_transcript(
                transcript=transcript,
                archetypal_cue_sentence=archetypal_cue_sentence,
            )

            # Clear any existing plots
            plt.close("all")
            _ = plt.figure()

            # Plot new
            sns.relplot(
                x="sentence_index",
                y="dist_sim",
                kind="line",
                data=all_transcript_sentence_distances,
            )

            # Save
            plt.savefig(
                PLOTS / transcript_path.with_suffix(".png").name, bbox_inches="tight"
            )

            # Save summary to CSV
            selected_breaks.to_csv(
                SUMMARIES
                / transcript_path.with_suffix(".csv").name.replace(
                    "transcript", "summary"
                ),
                index=False,
            )

        except (TypeError, KeyError) as e:
            print(
                f"Something wrong with transcript: {transcript_path}, skipping. "
                f"Error: {e}"
            )
