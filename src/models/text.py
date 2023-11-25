from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Tuple

import torch.nn as nn
import torch

from transformers import PreTrainedModel, GPT2LMHeadModel, GPT2Tokenizer, GPTNeoXForCausalLM, GPTNeoXTokenizerFast, LlamaForCausalLM, LlamaTokenizerFast, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from .base import VariableLengthClassifierOutput, VariableLengthModelForPrediction

class VariableLengthModelForPrediction(VariableLengthModelForPrediction):
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            # some models do not have a pad token
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
    
    @abc.abstractproperty
    def torso(self) -> nn.Module:
        """Main part of network"""
        ...
    
    @abc.abstractproperty
    def head(self) -> Optional[nn.Module]:
        """Last part of network"""
        ...
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None) -> VariableLengthClassifierOutput:
        # pass through model
        outputs: SequenceClassifierOutput = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        return VariableLengthClassifierOutput(
            layer_activations=outputs.hidden_states,
            predictions=outputs.logits,
        )
    
    @property
    def layers(self) -> List[Tuple[int, int]]:
        # no blocks in these models
        return [
            (i, 0) for i in range(len(self.torso) + 1)
        ]

    @classmethod
    def _from_pretrained(cls, model_name: str, **kwargs: Any) -> VariableLengthModelForPrediction:
        """Load a model from pretrained weights."""
        # use the built-in HuggingFace functionality
        return cls(AutoModelForCausalLM.from_pretrained(model_name, **kwargs, device_map="auto"), AutoTokenizer.from_pretrained(model_name))

    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # tokenize the sentences
        max_seq_length = 128
        return self.tokenizer(
            batch["text"],
            padding="max_length",
            max_length=max_seq_length,
            truncation=True,
        )

    @property
    def hf_device_map(self) -> Dict[str, torch.device]:
        return {
            f"model.{module}": device
            for module, device in self.model.hf_device_map.items()
        }


@VariableLengthModelForPrediction.register("gpt2")
class VariableLengthGPT2ForPrediction(VariableLengthModelForPrediction):
    def __init__(self, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer) -> None:
        super().__init__(model, tokenizer)
    
    @property
    def torso(self) -> nn.Module:
        return self.model.transformer.h

    @property
    def head(self) -> Optional[nn.Module]:
        return self.model.lm_head

@VariableLengthModelForPrediction.register("pythia")
class VariableLengthPythiaForPrediction(VariableLengthModelForPrediction):
    def __init__(self, model: GPTNeoXForCausalLM, tokenizer: GPTNeoXTokenizerFast) -> None:
        super().__init__(model, tokenizer)
    
    @property
    def torso(self) -> nn.Module:
        return self.model.gpt_neox.layers

    @property
    def head(self) -> Optional[nn.Module]:
        return self.model.embed_out


@VariableLengthModelForPrediction.register("llama-2")
class VariableLengthLlama2ForPrediction(VariableLengthModelForPrediction):
    def __init__(self, model: LlamaForCausalLM, tokenizer: LlamaTokenizerFast) -> None:
        super().__init__(model, tokenizer)

    @property
    def torso(self) -> nn.Module:
        return self.model.model.layers

    @property
    def head(self) -> Optional[nn.Module]:
        return self.model.lm_head
