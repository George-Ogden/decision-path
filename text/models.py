from transformers import BertForSequenceClassification


class ReducedLengthBert(BertForSequenceClassification):
    def reduce_layers(self, layers: int):
        self.bert.encoder.layer = self.bert.encoder.layer[:layers]

    @property
    def num_layers(self):
        return len(self.bert.encoder.layer)
