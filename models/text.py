from __future__ import annotations

import abc
from typing import Any, List, Optional, Tuple

import torch.nn as nn
from transformers import PreTrainedModel, BertForSequenceClassification, BertTokenizer, PreTrainedTokenizer, RobertaForSequenceClassification, RobertaTokenizer, GPT2ForSequenceClassification, GPT2Tokenizer, AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from .base import VariableLengthClassifierOutput, VariableLengthModelForClassification

class ReducedLengthModelForSequenceClassification(VariableLengthModelForClassification):
    @abc.abstractproperty
    def model(self) -> PreTrainedModel:
        ...
    
    @abc.abstractproperty
    def tokenizer(self) -> PreTrainedTokenizer:
        ...

    @abc.abstractproperty
    def torso(self) -> nn.Module:
        ...
    
    @abc.abstractproperty
    def head(self) -> Optional[nn.Module]:
        ...
    
    def forward(self, *args: Any, **kwargs: Any) -> VariableLengthClassifierOutput:
        kwargs |= {
            "output_hidden_states": True,
        }
        outputs: SequenceClassifierOutput = self.model(*args, **kwargs)
        predictions = None
        if self.head is not None:
            predictions = [self.head(hidden_state) for hidden_state in outputs.hidden_states]
        return VariableLengthClassifierOutput(
            layer_activations=outputs.hidden_states,
            layer_predictions=predictions,
        )
    
    @property
    def layers(self) -> List[Tuple[int, int]]:
        return [
            (i, 0) for i in range(len(self.torso) + 1)
        ]

    @classmethod
    def _from_pretrained(cls, model_name: str) -> ReducedLengthModelForSequenceClassification:
        return cls(AutoModelForSequenceClassification.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name))

@VariableLengthModelForClassification.register("bert")
class ReducedLengthBertForSequenceClassification(ReducedLengthModelForSequenceClassification):
    def __init__(self, model: BertForSequenceClassification, tokenizer: BertTokenizer) -> None:
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
    
    @property
    def tokenizer(self) -> BertTokenizer:
        return self._tokenizer

    @property
    def model(self) -> BertForSequenceClassification:
        return self._model
    
    @property
    def torso(self) -> nn.Module:
        return self.model.bert.encoder.layer
    
    @property
    def head(self) -> Optional[nn.Module]:
        return self.model.classifier

@VariableLengthModelForClassification.register("roberta")
class ReducedLengthRobertaForSequenceClassification(ReducedLengthModelForSequenceClassification):
    def __init__(self, model: RobertaForSequenceClassification, tokenizer: RobertaTokenizer) -> None:
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
    
    @property
    def tokenizer(self) -> RobertaTokenizer:
        return self._tokenizer

    @property
    def model(self) -> RobertaForSequenceClassification:
        return self._model
    
    @property
    def torso(self) -> nn.Module:
        return self.model.roberta.encoder.layer
    
    @property
    def head(self) -> Optional[nn.Module]:
        return self.model.classifier

@VariableLengthModelForClassification.register("gpt2")
class ReducedLengthGPT2ForSequenceClassification(ReducedLengthModelForSequenceClassification):
    def __init__(self, model: GPT2ForSequenceClassification, tokenizer: GPT2Tokenizer) -> None:
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
    
    @property
    def tokenizer(self) -> GPT2Tokenizer:
        return self._tokenizer

    @property
    def model(self) -> GPT2ForSequenceClassification:
        return self._model

    @property
    def torso(self) -> nn.Module:
        return self.model.transformer.h
    
    @property
    def head(self) -> Optional[nn.Module]:
        return self.model.score
