#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Top-level package for cue-queue."""

__author__ = "Jackson Maxfield Brown"
__email__ = "jmaxfieldbrown@gmail.com"

# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "4.1.0"

from .core import get_average_delimiter_encoding_for_corpus  # noqa: F401
from .core import get_average_delimiter_encoding_for_transcript  # noqa: F401


def get_module_version() -> str:
    return __version__
