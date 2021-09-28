#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import sys
import traceback

from cue_queue.core import train_from_corpus

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
            prog="cue-queue-train",
            description=(
                "Train a text cue sentence classifier model using the provided corpus."
            ),
        )
        p.add_argument(
            "corpus_uri",
            type=str,
            help="URI to annotated transcripts directory.",
        )
        p.add_argument(
            "-o",
            "--output_uri",
            type=str,
            default="cue-queue.pkl",
            help="Output URI to save the trained cue-queue model.",
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


def main() -> None:
    try:
        args = Args()
        train_from_corpus(
            corpus_uri=args.corpus_uri,
            output_uri=args.output_uri,
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
