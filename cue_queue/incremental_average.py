#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

###############################################################################


class IncrementalAverage:
    """
    Incremental Average manager.

    Useful when many numpy arrays are too large to fit in memory to average.

    Examples
    --------
    >>> inc_avg = IncrementalAverage()
    ... inc_avg.update(np.asarray([2]))
    ... np.testing.assert_array_equal(inc_avg.average, np.asarray([2]))
    ... inc_avg.update(np.asarray([3]))
    ... np.testing.assert_array_equal(inc_avg.average, np.asarray([2.5]))
    ... inc_avg.update(np.asarray([4]))
    ... np.testing.assert_array_equal(inc_avg.average, np.asarray([3]))
    """

    def __init__(self) -> None:
        self._average = None
        self._terms = 0

    @property
    def average(self) -> np.ndarray:
        """
        Get the current average.
        """
        if self._average is not None:
            return self._average

        raise ValueError("No terms have been added to the incremental average yet.")

    @property
    def terms(self) -> int:
        """
        Get the current number of terms included in the average.
        """
        return self._terms

    def update(self, addition: np.ndarray) -> None:
        """
        Add a term to the incremental average.
        """
        # Protect against None multiplication
        if self._average is None:
            average = 0
        else:
            average = self.average

        self._average = (average * self.terms + addition) / (self.terms + 1)
        self._terms += 1
