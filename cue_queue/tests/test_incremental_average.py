#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List

import numpy as np
import pytest

from cue_queue.incremental_average import IncrementalAverage

###############################################################################


@pytest.mark.parametrize(
    "terms, expected",
    [([np.asarray([2]), np.asarray([3]), np.asarray([4])], np.asarray([3]))],
)
def test_incremental_averaging(terms: List[np.ndarray], expected: np.ndarray) -> None:
    inc = IncrementalAverage()
    for term in terms:
        inc.update(term)

    np.testing.assert_array_equal(inc.average, expected)
