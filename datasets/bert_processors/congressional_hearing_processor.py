import os

from datasets.bert_processors.abstract_processor import BertProcessor, InputExample


class CongressionalHearingProcessor(BertProcessor):
    NUM_CLASSES = 6
    IS_MULTILABEL = True

    def __init__(self, config=None):
        super().__init__()
        self.use_text_b = config.use_second_input
        self.column_text_b = config.second_input_column
        if config.fold_num >= 0:
            self.NAME = os.path.join('CongressionalHearingFolds', 'fold'+str(config.fold_num))
        else:
            self.NAME = 'CongressionalHearing'

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, self.NAME, 'train.tsv')))

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, self.NAME, 'dev.tsv')))

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, self.NAME, 'test.tsv')))

    def _create_examples(self, lines):
        examples = []
        text_b = None
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = line[0]
            text_a = line[2]
            label = line[1]
            if self.use_text_b:
                text_b = line[self.second_input_column]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples