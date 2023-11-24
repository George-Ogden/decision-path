from .kurtosis import Kurtosis, RotatedKurtosis
from .outliers import Outliers
from .base import Metric
from .path import Path
from .rms import RMS

METRICS = Metric.registry