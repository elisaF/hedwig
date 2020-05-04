from torch import nn
from transformers import BertModel


class BertHierarchical(nn.Module):

    def __init__(self, model_name, num_fine_labels, num_coarse_labels):
        super().__init__()

        self.bert = BertModel.from_pretrained(model_name, num_labels=num_fine_labels)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier_coarse = nn.Linear(self.bert.config.hidden_size, num_coarse_labels)
        self.classifier_fine = nn.Linear(self.bert.config.hidden_size, num_fine_labels)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,):
        """
        a batch is a tensor of shape [batch_size, #file_in_commit, #line_in_file]
        and each element is a line, i.e., a bert_batch,
        which consists of input_ids, input_mask, segment_ids, label_ids
        """
        
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

        logits_coarse = self.classifier_coarse(pooled_output)
        logits_fine = self.classifier_fine(pooled_output)

        return logits_coarse, logits_fine
