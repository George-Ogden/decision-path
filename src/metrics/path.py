import numpy as np
import einops
import torch

from typing import Dict, List

from .base import Metric, VariableLengthClassifierOutput

@Metric.register("path")
class Path(Metric):
    """Path through residual stream."""
    def __init__(self):
        super().__init__()
        self.values = []
    
    def update(self, batch: Dict[str, torch.Tensor], model_output: VariableLengthClassifierOutput):
        super().update(batch, model_output)
        # L x [B, N, H]
        sequence_lengths = (torch.eq(batch["input_ids"], batch["pad_token_id"]).long().argmax(-1) - 1).to(
            model_output.layer_activations[0].device
        )

        self.values.append(
            [
                activations[torch.arange(batch["batch_size"], device=activations.device),sequence_lengths,:].detach().cpu().numpy()
                for activations in model_output.layer_activations
            ]
        )
        # NB x L x [B, H]
    
    def compute(self) -> List[List[float]]:
        return np.stack([
            # [NB, B, H] -> [N, H]
            np.concatenate(layer, axis=0)
            for layer in zip(*self.values)
        ]).transpose(1, 0, 2)