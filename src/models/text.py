from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Tuple

import torch.nn as nn
import torch

from transformers import PreTrainedModel, BertForSequenceClassification, BertTokenizer, PreTrainedTokenizer, RobertaForSequenceClassification, RobertaTokenizer, GPT2ForSequenceClassification, GPT2Tokenizer, AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from .base import VariableLengthClassifierOutput, VariableLengthModelForClassification

class VariableLengthModelForSequenceClassification(VariableLengthModelForClassification):
    def __init__(self, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    @abc.abstractproperty
    def torso(self) -> nn.Module:
        ...
    
    @abc.abstractproperty
    def head(self) -> Optional[nn.Module]:
        ...
    
    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor) -> VariableLengthClassifierOutput:
        outputs: SequenceClassifierOutput = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
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
    def _from_pretrained(cls, model_name: str) -> VariableLengthModelForSequenceClassification:
        return cls(AutoModelForSequenceClassification.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name))

    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        sentence1_key, sentence2_key = "premise", "hypothesis"
        max_seq_length = 128
        return self.tokenizer(
            batch[sentence1_key],
            batch[sentence2_key],
            padding="max_length",
            max_length=max_seq_length,
            truncation=True,
        )

@VariableLengthModelForClassification.register("bert")
class VariableLengthBertForSequenceClassification(VariableLengthModelForSequenceClassification):
    def __init__(self, model: BertForSequenceClassification, tokenizer: BertTokenizer) -> None:
        super().__init__(model, tokenizer)   
    
    @property
    def torso(self) -> nn.Module:
        return self.model.bert.encoder.layer
    
    @property
    def head(self) -> Optional[nn.Module]:
        return self.model.classifier

@VariableLengthModelForClassification.register("roberta")
class VariableLengthRobertaForSequenceClassification(VariableLengthModelForSequenceClassification):
    def __init__(self, model: RobertaForSequenceClassification, tokenizer: RobertaTokenizer) -> None:
        super().__init__(model, tokenizer)
    
    @property
    def torso(self) -> nn.Module:
        return self.model.roberta.encoder.layer
    
    @property
    def head(self) -> Optional[nn.Module]:
        return self.model.classifier

@VariableLengthModelForClassification.register("gpt2")
class VariableLengthGPT2ForSequenceClassification(VariableLengthModelForSequenceClassification):
    def __init__(self, model: GPT2ForSequenceClassification, tokenizer: GPT2Tokenizer) -> None:
        super().__init__(model, tokenizer)
        if self.tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

    @property
    def torso(self) -> nn.Module:
        return self.model.transformer.h
    
    @property
    def head(self) -> Optional[nn.Module]:
        return self.model.score
