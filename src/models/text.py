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
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> VariableLengthClassifierOutput:
        # pass through model
        outputs: SequenceClassifierOutput = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        predictions = None
        if self.head is not None:
            # make predictions if head is present
            predictions = [self.head(hidden_state) for hidden_state in outputs.hidden_states]
        return VariableLengthClassifierOutput(
            layer_activations=outputs.hidden_states,
            layer_predictions=predictions,
        )
    
    @property
    def layers(self) -> List[Tuple[int, int]]:
        # no blocks in these models
        return [
            (i, 0) for i in range(len(self.torso) + 1)
        ]

    @classmethod
    def _from_pretrained(cls, model_name: str) -> VariableLengthModelForSequenceClassification:
        """Load a model from pretrained weights."""
        # use the built-in HuggingFace functionality
        return cls(AutoModelForSequenceClassification.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name))

    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # tokenize the sentences
        question = "question"
        max_seq_length = 128
        return self.tokenizer(
            batch[question],
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
            # GPT2 does not have a pad token
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

    @property
    def torso(self) -> nn.Module:
        return self.model.transformer.h
   
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> VariableLengthClassifierOutput:
        outputs: VariableLengthClassifierOutput = super().forward(input_ids=input_ids, attention_mask=attention_mask)
        # compute predictions based on lengths
        # modified from https://github.com/huggingface/transformers/blob/b257c46a075419c09e5ce5c5aa39bc346ecdb9a5/src/transformers/models/gpt2/modeling_gpt2.py#L1446
        sequence_lengths = (torch.ne(input_ids, self.model.config.pad_token_id).sum(dim=-1) - 1)
        hidden_states = torch.stack([
            activations[torch.arange(activations.shape[0], device=activations.device), sequence_lengths.to(activations.device)]
            for activations in outputs.layer_activations
        ])
        predictions = self.model.score(hidden_states)
        return VariableLengthClassifierOutput(
            layer_activations=outputs.layer_activations,
            layer_predictions=predictions,
        ) 
    @property
    def head(self) -> Optional[nn.Module]:
        return None

class RMSLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-8, affine=True):
        super(RMSLayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(()))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_normalized = x / rms
        if self.affine:
            x_normalized = x_normalized * self.weight
        return x_normalized

def replace(model):
    for name, child in model.named_children():
        if isinstance(child, nn.modules.normalization.LayerNorm):
            setattr(model, name, RMSLayerNorm(child.normalized_shape, eps=child.eps, affine=True))
        else:
            replace(child)
    return model


class GPTR2ForSequenceClassification(GPT2ForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        replace(self)

@VariableLengthModelForClassification.register("gptr2")
class VariableLengthGPTR2ForSequenceClassification(VariableLengthGPT2ForSequenceClassification):
    @classmethod
    def _from_pretrained(cls, model_name: str) -> VariableLengthModelForSequenceClassification:
        return cls(GPTR2ForSequenceClassification.from_pretrained(model_name), GPT2Tokenizer.from_pretrained(model_name))

