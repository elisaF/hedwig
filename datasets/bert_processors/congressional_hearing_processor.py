import os

from datasets.bert_processors.abstract_processor import BertProcessor, InputExample


class CongressionalHearingProcessor(BertProcessor):
    NAME = 'CongressionalHearing'
    NUM_CLASSES = 6
    IS_MULTILABEL = True

    def __init__(self, config=None):
        super().__init__()
        self.use_text_b = config.use_text_b

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'CongressionalHearing', 'train.tsv')))

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'CongressionalHearing', 'dev.tsv')))

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'CongressionalHearing', 'test.tsv')))

    def _create_examples(self, lines):
        examples = []
        text_b = None
        for (i, line) in enumerate(lines):
            guid = line[0]
            text_a = line[2]
            label = line[1]
            if self.use_text_b:
                text_b = line[3]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples