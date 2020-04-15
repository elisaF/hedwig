import os

from datasets.bow_processors.abstract_processor import BagOfWordsProcessor, InputExample


class CongressionalHearingProcessor(BagOfWordsProcessor):
    NAME = 'CongressionalHearing'
    NUM_CLASSES = 6
    VOCAB_SIZE = 36308  ## change??
    IS_MULTILABEL = True

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
        for (i, line) in enumerate(lines):
            examples.append(InputExample(guid=line[0], text=line[2], label=line[1]))
        return examples
