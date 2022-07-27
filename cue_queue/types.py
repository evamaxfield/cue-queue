#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dataclasses_json import dataclass_json

###############################################################################


@dataclass_json
@dataclass
class Sentence:
    text: str
    index: int
    metadata: Optional[Dict[str, Any]] = None


@dataclass_json
@dataclass
class Section:
    name: str
    start_index: int
    description: Optional[str] = None


@dataclass_json
@dataclass
class Transcript:
    sentences: List[Sentence]
    sections: Optional[List[Section]] = None
    metadata: Optional[Dict[str, Any]] = None
