#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict, List, NamedTuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cdp_backend.pipeline.transcript_model import (
    SectionAnnotation,
    Transcript,
    TranscriptAnnotations,
)
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sns.set_theme(style="darkgrid")

###############################################################################
# Globals and Model Init

ANNOTATED_DATASET = Path("annotated").resolve(strict=True)
PLOTS = Path("plots").resolve()
PLOTS.mkdir(exist_ok=True)
SUMMARIES = Path("summaries").resolve()
SUMMARIES.mkdir(exist_ok=True)

TRANSFORMER = SentenceTransformer(
    "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
)

###############################################################################


def get_full_section_description(anno: SectionAnnotation) -> str:
    if anno.description is not None:
        return f"{anno.name}, {anno.description}"

    return f"{anno.name}"


class SectionDetails(NamedTuple):
    name: str
    seed: str


class EmbeddedSectionDetails(NamedTuple):
    details: SectionDetails
    embedding: np.ndarray


def process_transcript(
    section_details: List[SectionDetails], transcript: Transcript
) -> pd.DataFrame:
    # Get all section embeddings
    embedded_sections: List[EmbeddedSectionDetails] = []
    for details in section_details:
        embedded_sections.append(
            EmbeddedSectionDetails(
                details=details,
                embedding=TRANSFORMER.encode(details.seed),
            )
        )

    # Process all sentences and store section data as
    # [{"sentence_index": 1, "section_name": "CB 1111", "similarity": 11.51}, ... ]
    section_data: List[Dict[str, Union[int, str, np.float32]]] = []

    # Safety measure to ensure the transcript sentences are in sorted order
    for sentence in tqdm(
        sorted(transcript.sentences, key=lambda s: s.index), "Sentences processed"
    ):
        # Get this sentence embedding and get distance from each section embedding
        sentence_embedding = TRANSFORMER.encode(sentence.text)
        for section_embedding in embedded_sections:
            section_data.append(
                {
                    "sentence_index": sentence.index,
                    "section_name": section_embedding.details.name,
                    "distance": np.linalg.norm(
                        section_embedding.embedding - sentence_embedding
                    ),
                }
            )

    return pd.DataFrame(section_data)


###############################################################################

if __name__ == "__main__":
    # Read and process each transcript
    for transcript_path in tqdm(
        list(ANNOTATED_DATASET.glob("*-transcript.json")), "Transcripts processed"
    ):
        try:
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

            # Process
            transcript_results = process_transcript(
                section_details=[
                    SectionDetails(
                        name=section.name,
                        seed=get_full_section_description(section),
                    )
                    for section in sections
                ],
                transcript=transcript,
            )

            # Clear any existing plots
            plt.close("all")
            _ = plt.figure()

            # Plot new
            sns.relplot(
                x="sentence_index",
                y="distance",
                hue="section_name",
                kind="line",
                data=transcript_results,
            )

            # Save
            plt.savefig(
                PLOTS / transcript_path.with_suffix(".png").name, bbox_inches="tight"
            )

            # Summarize
            summarized_transcript_results_list: List[
                Dict[str, Union[str, int, np.float32]]
            ] = []
            for section in sections:
                section_min_distance_idx = transcript_results[
                    transcript_results.section_name == section.name
                ].distance.idxmin()
                section_min_distance_details = transcript_results.loc[
                    section_min_distance_idx
                ]
                summarized_transcript_results_list.append(
                    {
                        "section_name": section.name,
                        "true_section_start": section.start_sentence_index,
                        "true_section_end": section.end_sentence_index,
                        "predicted_section_min_distance_sentence_idx": (
                            section_min_distance_details.sentence_index
                        ),
                        "predicted_section_min_distance_sentence_text": (
                            transcript.sentences[
                                section_min_distance_details.sentence_index
                            ].text
                        ),
                        "predicted_section_min_distance": (
                            section_min_distance_details.distance
                        ),
                    }
                )

            # Save summary to CSV
            summarized_transcript_results = pd.DataFrame(summarized_transcript_results_list)
            summarized_transcript_results.to_csv(
                SUMMARIES
                / transcript_path.with_suffix(".csv").name.replace("transcript", "summary"),
                index=False,
            )

        except (TypeError, KeyError) as e:
            print(
                f"Something wrong with transcript: {transcript_path}, skipping. "
                f"Error: {e}"
            )
