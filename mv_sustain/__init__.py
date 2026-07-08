"""
mv-sustain: a longitudinal, multi-visit extension of the SuStaIn algorithm.

Built on top of pySuStaIn (Aksman, Wijeratne, et al.; https://github.com/ucl-pond/pySuStaIn), the reference implementation of SuStaIn (Young et al. 2018, Nat Commun), this package adds a joint patient-level likelihood so that repeated visits from the same patient reinforce a single subtype/stage estimate instead of being scored as independent, unrelated observations.

Two model families are provided for each supported likelihood (z-score and ordinal):

- Stacked* classes: classic, independent-visit SuStaIn (each visit trained as its own trivial patient). Provided as the fair baseline for comparison.
- Longitudinal* classes: the joint patient-level likelihood extension (MV-SuStaIn, "Multi-Visit SuStaIn").

`SustainRunner` (in `sustain_utils`) is the recommended entry point — it routes to the correct class given a likelihood type and a `use_longitudinal_likelihood` flag, so most users should not need to instantiate the model classes directly.
"""

from .sustain_utils import SustainRunner, Likelihood
from .stacked_sustain_override import StackedZscoreSustain, StackedOrdinalSustain
from .longitudinal_override import (
    LongitudinalZscoreSustain,
    LongitudinalOrdinalSustain,
    LongitudinalMixtureSustain,
)
from .zscore_override import ZscoreSustain
from .ordinal_override import OrdinalSustain
from .mixture_override import MixtureSustain

__version__ = "0.1.3"

__all__ = [
    "SustainRunner",
    "Likelihood",
    "StackedZscoreSustain",
    "StackedOrdinalSustain",
    "LongitudinalZscoreSustain",
    "LongitudinalOrdinalSustain",
    "LongitudinalMixtureSustain",
    "ZscoreSustain",
    "OrdinalSustain",
    "MixtureSustain",
]
