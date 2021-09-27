#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import sys
import traceback

import numpy as np
from fsspec.core import url_to_fs

from cue_queue import get_average_cue_sentence_encoding_for_corpus

###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",
)
log = logging.getLogger(__name__)

###############################################################################


class Args(argparse.Namespace):
    def __init__(self) -> None:
        self.__parse()

    def __parse(self) -> None:
        p = argparse.ArgumentParser(
            prog="get-average-cue-sentence-encoding-for-corpus",
            description=(
                "Process annotated transcripts, generate an average cue sentence "
                "encoding, and store to numpy array."
            ),
        )
        p.add_argument(
            "annotated_transcripts_dir",
            type=str,
            help="URI to annotated transcripts directory.",
        )
        p.add_argument(
            "-o",
            "--output_path",
            type=str,
            default="average-cue-sentence.npy",
            help=(
                "Output path to save the average cue sentence encoding to. "
                "Must be local."
            ),
        )
        p.add_argument(
            "-s",
            "--strict",
            action="store_true",
            help=(
                "When provided, will raise on any transcript error. "
                "Default: False (skip problematic transcripts)"
            ),
        )
        p.parse_args(namespace=self)


###############################################################################


def _get_average_cue_sentence_encoding_for_corpus(
    annotated_transcripts_dir: str,
    output_path: str,
    strict: bool = False,
) -> None:
    # Get fs and path specification
    fs, path = url_to_fs(annotated_transcripts_dir)

    # Generate average sentence encoding
    average_encoding = get_average_cue_sentence_encoding_for_corpus(
        transcripts=fs.ls(path),
        strict=strict,
    )

    # Save
    np.save(output_path, average_encoding)


def main() -> None:
    try:
        args = Args()
        _get_average_cue_sentence_encoding_for_corpus(
            annotated_transcripts_dir=args.annotated_transcripts_dir,
            output_path=args.output_path,
            strict=args.strict,
        )
    except Exception as e:
        log.error("=============================================")
        log.error("\n\n" + traceback.format_exc())
        log.error("=============================================")
        log.error("\n\n" + str(e) + "\n")
        log.error("=============================================")
        sys.exit(1)


###############################################################################
# Allow caller to directly run this module (usually in development scenarios)

if __name__ == "__main__":
    main()
