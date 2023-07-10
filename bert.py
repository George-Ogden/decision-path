from transformers.models.bert import BertForSequenceClassification

class ReducedLengthBert(BertForSequenceClassification):
    def post_init(self):
        super().post_init()
        self.bert.encoder.layer = self.bert.encoder.layer[:11]

def main():
    model = ReducedLengthBert.from_pretrained("mnli")
    print(model.bert.encoder.layer[:11])

if __name__ == "__main__":
    main()