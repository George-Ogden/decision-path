import numpy as np
import torch

from typing import Dict

from .base import Metric, VariableLengthClassifierOutput

@Metric.register("topk")
class Topk(Metric):
    def __init__(self, k: int):
        self.value = 0
        self.k = k
        super().__init__()

    def update(self, batch: Dict[str, torch.Tensor], model_output: VariableLengthClassifierOutput):
        # L x [B, C]
        self.value += np.sum(
            [
                (torch.topk(prediction, k=self.k, dim=-1).indices == batch["labels"].unsqueeze(-1)).sum().item()
                for prediction in model_output.predictions
            ],
            axis=1
        )
        super().update(batch, model_output)

@Metric.register("accuracy")
@Metric.register("top1")
class Top1(Topk):
    def __init__(self):
        super().__init__(k=1)

@Metric.register("top5")
class Top5(Topk):
    def __init__(self):
        super().__init__(k=5)