#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import List, Optional, Union

import fireo
from gcsfs import GCSFileSystem
from cdp_backend.database import models as db_models
from cdp_backend.pipeline import transcript_model as cdp_transcript_types
from google.auth.credentials import AnonymousCredentials
from google.cloud.firestore import Client
from tqdm import tqdm

from .. import types as qq_types

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def generate_dataset(
    save_dir: Optional[Union[str, Path]] = None,
    clean: bool = False,
    instance: str = "cdp-seattle-staging-dbengvtn",
    n_transcripts: int = 50,
) -> Path:
    """
    Generate a cue-queue dataset from a CDP instance's transcripts.

    Parameters
    ----------
    save_dir: Optional[Union[str, Path]]
        Which directory to save the generated dataset to.
        Default: None (use the provided instance string as the save directory name)
    clean: bool = False
        Should the save directory be cleaned before saving new files to it.
        Default: False (do not clean)
    instance: str
        Which CDP instance to request data from.
        Default: seattle-staging ("cdp-seattle-staging-dbengvtn")
    n_transcripts: int
        Number of transcripts to generate.
        For CDP this is equal to the number of events to query.
        Default: 50

    Returns
    -------
    dataset_path: Path
        The local path where all converted transcripts are stored.
    """
    # Try the instance before doing any path handling
    # Connect to the db and fs
    fireo.connection(
        client=Client(project=instance, credentials=AnonymousCredentials())
    )
    fs = GCSFileSystem(project="cdp-seattle-staging-dbengvtn", token="anon")

    # Get events
    # This could raise a PermissionDenied error or similar if the instance doesn't exist
    # Or has private settings / access
    events = list(db_models.Event.collection.fetch(n_transcripts))

    # Check save dir
    if save_dir is None:
        save_dir = f"corpus--{instance}"
    save_dir = Path(save_dir).resolve()
    if save_dir.is_file():
        raise FileExistsError(
            f"Provided save directory already exists as a file: '{save_dir}"
        )

    log.info(f"Generated dataset will be stored to: {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)
    if clean:
        log.info("Cleaning existing save directory of files before beginning.")
        for f in save_dir.glob("*"):
            f.unlink()

    # Get each event's sessions and each session's highest confidence transcript
    # Combine each event's multiple transcripts into a single cue-queue transcript
    #
    # Basically, in CDPs model,
    # an event can have multiple sessions and the minutes items are split across those
    # sessions. So to prepare the dataset for cue-queue we need to combine the multiple
    # sessions into a single object
    for event in tqdm(events, desc="CDP Events Processed"):
        qq_transcript_sentences: List[qq_types.Sentence] = []
        sentence_counter = 0

        sessions = list(
            db_models.Session.collection.order("session_index")
            .filter("event_ref", "==", event.key)
            .fetch()
        )
        for session in sessions:
            transcript_db_model = (
                db_models.Transcript.collection.order("-confidence")
                .filter("session_ref", "==", session.key)
                .get()
            )
            # Read the transcript
            with fs.open(transcript_db_model.file_ref.get().uri, "r") as open_resource:
                cdp_transcript = (
                    cdp_transcript_types.Transcript.from_json(  # type: ignore
                        open_resource.read()
                    )
                )

            # Convert the sentences and track any session splits
            for cdp_sentence in sorted(cdp_transcript.sentences, key=lambda s: s.index):
                qq_transcript_sentences.append(
                    qq_types.Sentence(
                        text=cdp_sentence.text,
                        index=sentence_counter,
                        metadata={"session_id": session.id},
                    )
                )
                sentence_counter += 1

        # Attach minutes items
        event_minutes_items = list(
            db_models.EventMinutesItem.collection.order("index")
            .filter("event_ref", "==", event.key)
            .fetch()
        )

        minutes_items = [emi.minutes_item_ref.get() for emi in event_minutes_items]

        # Construct final object
        qq_transcript = qq_types.Transcript(
            sentences=qq_transcript_sentences,
            metadata={
                "event_id": event.id,
                "minutes_items": [
                    {
                        "name": mi.name,
                        "description": mi.description,
                    }
                    for mi in minutes_items
                ],
            },
        )

        # Store to dataset save dir
        with open(save_dir / f"{event.id}.json", "w") as open_resource:
            open_resource.write(qq_transcript.to_json())  # type: ignore

    return save_dir
