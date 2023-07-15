from transformers import BertForSequenceClassification, RobertaForSequenceClassification


class ReducedLengthBert(BertForSequenceClassification):
    def reduce_layers(self, layers: int):
        self.bert.encoder.layer = self.bert.encoder.layer[:layers]

    @property
    def num_layers(self):
        return len(self.bert.encoder.layer)


class ReducedLengthRoberta(RobertaForSequenceClassification):
    def reduce_layers(self, layers: int):
        self.roberta.encoder.layer = self.roberta.encoder.layer[:layers]

    @property
    def num_layers(self):
        return len(self.roberta.encoder.layer)