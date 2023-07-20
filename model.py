from dataclasses import dataclass
from typing import Optional
import torch
from transformers.modeling_outputs import SequenceClassifierOutput


@dataclass
class VariableLengthSequenceClassifierOutput(SequenceClassifierOutput):
    layer_predictions: Optional[torch.FloatTensor] = None