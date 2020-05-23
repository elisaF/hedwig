from torch import nn
from transformers import ElectraModel


class ElectraForSequenceClassification(nn.Module):

    def __init__(self, model_name, num_labels):
        super().__init__()
        self.num_labels = num_labels

        self.electra = ElectraModel.from_pretrained(model_name, num_labels=num_labels)
        self.dropout = nn.Dropout(self.electra.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.electra.config.hidden_size, self.electra.config.num_labels)

        self.electra.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

