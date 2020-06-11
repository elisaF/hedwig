import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm

from datasets.bert_processors.abstract_processor import convert_examples_to_features, \
    convert_examples_to_hierarchical_features
from utils.preprocessing import pad_input_matrix, get_coarse_labels

# Suppress warnings from sklearn.metrics
warnings.filterwarnings('ignore')


class BertEvaluator(object):
    def __init__(self, model, processor, tokenizer, args, split='dev'):
        self.args = args
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer

        if split == 'test':
            self.eval_examples = self.processor.get_test_examples(args.data_dir)
        else:
            self.eval_examples = self.processor.get_dev_examples(args.data_dir)

    def get_scores(self, silent=False):
        if self.args.is_hierarchical:
            eval_features = convert_examples_to_hierarchical_features(
                self.eval_examples, self.args.max_seq_length, self.tokenizer)
        else:
            eval_features = convert_examples_to_features(
                self.eval_examples, self.args.max_seq_length, self.tokenizer, use_guid=True, is_regression=self.args.is_regression)

        unpadded_input_ids = [f.input_ids for f in eval_features]
        unpadded_input_mask = [f.input_mask for f in eval_features]
        unpadded_segment_ids = [f.segment_ids for f in eval_features]

        if self.args.is_hierarchical:
            pad_input_matrix(unpadded_input_ids, self.args.max_doc_length)
            pad_input_matrix(unpadded_input_mask, self.args.max_doc_length)
            pad_input_matrix(unpadded_segment_ids, self.args.max_doc_length)

        padded_input_ids = torch.tensor(unpadded_input_ids, dtype=torch.long)
        padded_input_mask = torch.tensor(unpadded_input_mask, dtype=torch.long)
        padded_segment_ids = torch.tensor(unpadded_segment_ids, dtype=torch.long)
        if self.args.is_regression:
            label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)
        else:
            label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        doc_ids = torch.tensor([f.guid for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(padded_input_ids, padded_input_mask, padded_segment_ids, label_ids, doc_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.args.batch_size)

        self.model.eval()

        total_loss = 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predicted_labels, target_labels, target_doc_ids = list(), list(), list()

        for input_ids, input_mask, segment_ids, label_ids, doc_ids in tqdm(eval_dataloader, desc="Evaluating",
                                                                           disable=silent):
            input_ids = input_ids.to(self.args.device)
            input_mask = input_mask.to(self.args.device)
            segment_ids = segment_ids.to(self.args.device)
            label_ids = label_ids.to(self.args.device)
            target_doc_ids.extend(doc_ids.tolist())

            with torch.no_grad():
                logits = self.model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)[0]

            if self.args.is_multilabel:
                predicted_labels.extend(F.sigmoid(logits).round().long().cpu().detach().numpy())
                target_labels.extend(label_ids.cpu().detach().numpy())
                # if self.args.pos_weights:
                #     pos_weights = [float(w) for w in self.args.pos_weights.split(',')]
                #     pos_weight = torch.FloatTensor(pos_weights)
                # else:
                #     pos_weight = torch.ones([self.args.num_labels])
                if self.args.loss == 'cross-entropy':
                    criterion = torch.nn.BCEWithLogitsLoss(size_average=False)
                    loss = criterion(logits.cpu(), label_ids.float().cpu())
                elif self.args.loss == 'mse':
                    criterion = torch.nn.MSELoss(size_average=False)
                    m = torch.nn.Sigmoid()
                    loss = criterion(m(logits.cpu()), label_ids.float().cpu())
            else:
                predicted_labels.extend(torch.argmax(logits, dim=1).cpu().detach().numpy())
                target_labels.extend(label_ids.cpu().detach().numpy())
                if self.args.num_labels > 2:
                    loss = F.cross_entropy(logits, torch.argmax(label_ids, dim=1))
                else:
                    if self.args.loss == 'mse' or self.args.is_regression:
                        criterion = torch.nn.MSELoss(size_average=False)
                        m = torch.nn.Sigmoid()
                        loss = criterion(m(logits.cpu()), label_ids.float().cpu())
                    elif self.args.loss == 'cross-entropy':
                        loss_fct = torch.nn.CrossEntropyLoss()
                        loss = loss_fct(logits.view(-1, self.args.num_labels), label_ids.view(-1))

            if self.args.n_gpu > 1:
                loss = loss.mean()
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            total_loss += loss.item()

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        avg_loss = total_loss / nb_eval_steps

        if self.args.is_regression:
            rmse = np.sqrt(metrics.mean_squared_error(target_labels, predicted_labels))

            score_values = [rmse,
                            avg_loss,
                            list(zip(target_doc_ids, target_labels, predicted_labels))]
            score_names = ['rmse',
                           'avg_loss',
                           'label_set_info (id/gold/pred)']

        else:
            hamming_loss = metrics.hamming_loss(target_labels, predicted_labels)

            predicted_labels, target_labels = np.array(predicted_labels), np.array(target_labels)
            cm = metrics.multilabel_confusion_matrix(target_labels, predicted_labels)
            accuracy = metrics.accuracy_score(target_labels, predicted_labels)

            if self.args.num_labels == 2:
                precision = metrics.precision_score(target_labels, predicted_labels, average='binary')
                recall = metrics.recall_score(target_labels, predicted_labels, average='binary')
                f1 = metrics.f1_score(target_labels, predicted_labels, average='binary')
            else:
                precision_micro = metrics.precision_score(target_labels, predicted_labels, average='micro')
                recall_micro = metrics.recall_score(target_labels, predicted_labels, average='micro')
                f1_micro = metrics.f1_score(target_labels, predicted_labels, average='micro')

                precision_macro = metrics.precision_score(target_labels, predicted_labels, average='macro')
                recall_macro = metrics.recall_score(target_labels, predicted_labels, average='macro')
                f1_macro = metrics.f1_score(target_labels, predicted_labels, average='macro')

                precision_class, recall_class, f1_class, support_class = metrics.precision_recall_fscore_support(
                    target_labels,
                    predicted_labels)

            predicted_label_sets = [predicted_label.tolist() for predicted_label in predicted_labels]
            target_label_sets = [target_label.tolist() for target_label in target_labels]

            if self.args.num_labels == 2:
                score_values = [precision, recall, f1,
                                accuracy,
                                avg_loss,
                                hamming_loss,
                                cm.tolist(), list(zip(target_doc_ids, target_label_sets, predicted_label_sets))]
                score_names = ['precision', 'recall', 'f1',
                               'accuracy',
                               'avg_loss',
                               'hamming_loss',
                               'confusion_matrix', 'label_set_info (id/gold/pred)']
            else:
                score_values = [precision_macro, recall_macro, f1_macro,
                                accuracy,
                                avg_loss,
                                hamming_loss,
                                precision_micro, recall_micro, f1_micro,
                                precision_class.tolist(), recall_class.tolist(), f1_class.tolist(), support_class.tolist(),
                                cm.tolist(), list(zip(target_doc_ids, target_label_sets, predicted_label_sets))]
                score_names = ['precision_macro', 'recall_macro', 'f1_macro',
                               'accuracy',
                               'avg_loss',
                               'hamming_loss',
                               'precision_micro', 'recall_micro', 'f1_micro',
                               'precision_class', 'recall_class', 'f1_class', 'support_class',
                               'confusion_matrix', 'label_set_info (id/gold/pred)']
        return score_values, score_names
