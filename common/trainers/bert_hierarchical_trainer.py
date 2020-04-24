import datetime
import os
import time

import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm
from tqdm import trange

from common.evaluators.bert_evaluator import BertEvaluator
from datasets.bert_processors.abstract_processor import convert_examples_to_features
from datasets.bert_processors.abstract_processor import convert_examples_to_hierarchical_features
from utils.preprocessing import pad_input_matrix, get_coarse_labels


class BertHierarchicalTrainer(object):
    def __init__(self, model_coarse, model_fine, optimizer, processor, scheduler, tokenizer, args):
        self.args = args
        self.model_coarse = model_coarse
        self.model_fine = model_fine
        self.optimizer_coarse = optimizer
        self.optimizer_fine = optimizer
        self.processor = processor
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.train_examples = self.processor.get_train_examples(args.data_dir)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.snapshot_path_coarse = os.path.join(self.args.save_path, self.processor.NAME+'_coarse', '%s.pt' % timestamp)
        self.snapshot_path_fine = os.path.join(self.args.save_path, self.processor.NAME+'_fine', '%s.pt' % timestamp)

        self.num_train_optimization_steps = int(
            len(self.train_examples) / args.batch_size / args.gradient_accumulation_steps) * args.epochs

        self.log_header = 'Epoch Iteration Progress   Dev/Acc.  Dev/Pr.  Dev/Re.   Dev/F1   Dev/Loss'
        self.log_template = ' '.join('{:>5.0f},{:>9.0f},{:>6.0f}/{:<5.0f} {:>6.4f},{:>8.4f},{:8.4f},{:8.4f},{:10.4f}'.split(','))

        self.iterations, self.nb_tr_steps, self.tr_loss_coarse, self.tr_loss_fine = 0, 0, 0, 0
        self.best_dev_f1, self.unimproved_iters = 0, 0
        self.early_stop = False

    def train_epoch(self, train_dataloader):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            self.model_coarse.train()
            self.model_fine.train()
            batch = tuple(t.to(self.args.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            logits_coarse = self.model_coarse(input_ids, input_mask, segment_ids)[0] # batch-size, num_classes
            logits_fine = self.model_fine(input_ids, input_mask, segment_ids)[0]

            # get coarse labels from the fine labels
            label_ids_coarse = get_coarse_labels(label_ids, self.args.num_coarse_labels,
                                                 self.args.parent_to_child_index_map, 
                                                 self.args.device)
            
            # calculate weights to ignore invalid 
            # fine labels based on gold coarse labels
            fine_loss_weights = self.get_coarse_weights(label_ids_coarse)

            if self.args.loss == 'cross-entropy':
                if self.args.pos_weights_coarse:
                    pos_weights_coarse = [float(w) for w in self.args.pos_weights_coarse.split(',')]
                    pos_weight_coarse = torch.FloatTensor(pos_weights_coarse)
                else:
                    pos_weight_coarse = torch.ones([self.args.num_coarse_labels])
                if self.args.pos_weights:
                    pos_weights = [float(w) for w in self.args.pos_weights.split(',')]
                    pos_weights = torch.FloatTensor(pos_weights)
                else:
                    pos_weights = torch.ones([self.args.num_labels])

                criterion_coarse = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_coarse)
                criterion_coarse = criterion_coarse.to(self.args.device)
                loss_coarse = criterion_coarse(logits_coarse, label_ids_coarse.float())

                criterion_fine = torch.nn.BCEWithLogitsLoss(weight=fine_loss_weights, pos_weight=pos_weights)
                criterion_fine = criterion_fine.to(self.args.device)
                loss_fine = criterion_fine(logits_fine, label_ids.float())

            loss_coarse.backward()
            loss_fine.backward()
            self.tr_loss_coarse += loss_coarse.item()
            self.tr_loss_fine += loss_fine.item()
            self.nb_tr_steps += 1
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                self.optimizer_coarse.step()
                self.optimizer_fine.step()
                self.scheduler.step()
                self.optimizer_coarse.zero_grad()
                self.optimizer_fine.zero_grad()
                self.iterations += 1

    def get_coarse_weights(self, gold_coarse_labels):
        weights = []
        for parent_idx, child_idxs in self.args.parent_to_child_index_map.items():
            weights.append(gold_coarse_labels[:, parent_idx].bool().repeat(len(child_idxs), 1).transpose(0, 1))
        weight = torch.cat(weights, 1)
        return weight.float()

    def train(self):
        if self.args.is_hierarchical:
            train_features = convert_examples_to_hierarchical_features(
                self.train_examples, self.args.max_seq_length, self.tokenizer)
        else:
            train_features = convert_examples_to_features(
                self.train_examples, self.args.max_seq_length, self.tokenizer, use_guid=True)

        unpadded_input_ids = [f.input_ids for f in train_features]
        unpadded_input_mask = [f.input_mask for f in train_features]
        unpadded_segment_ids = [f.segment_ids for f in train_features]

        if self.args.is_hierarchical:
            pad_input_matrix(unpadded_input_ids, self.args.max_doc_length)
            pad_input_matrix(unpadded_input_mask, self.args.max_doc_length)
            pad_input_matrix(unpadded_segment_ids, self.args.max_doc_length)

        print("Number of examples: ", len(self.train_examples))
        print("Batch size:", self.args.batch_size)
        print("Num of steps:", self.num_train_optimization_steps)

        padded_input_ids = torch.tensor(unpadded_input_ids, dtype=torch.long)
        padded_input_mask = torch.tensor(unpadded_input_mask, dtype=torch.long)
        padded_segment_ids = torch.tensor(unpadded_segment_ids, dtype=torch.long)
        label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(padded_input_ids, padded_input_mask, padded_segment_ids, label_ids)

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.args.batch_size)

        print('Begin training: ', datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        start_time = time.monotonic()
        for epoch in trange(int(self.args.epochs), desc="Epoch"):
            self.train_epoch(train_dataloader)
            if self.args.evaluate_dev:
                dev_evaluator_coarse = BertEvaluator(self.model_coarse, self.processor,
                                                     self.tokenizer, self.args, split='dev', map_labels=True)
                dev_evaluator_fine = BertEvaluator(self.model_fine, self.processor,
                                                   self.tokenizer, self.args, split='dev')
                dev_precision_coarse, dev_recall_coarse, dev_f1_coarse, dev_acc_coarse, dev_loss_coarse = \
                    dev_evaluator_coarse.get_scores()[0][:5]
                dev_precision_fine, dev_recall_fine, dev_f1_fine, dev_acc_fine, dev_loss_fine = \
                    dev_evaluator_fine.get_scores()[0][:5]

                # Print validation results
                tqdm.write('COARSE: '+self.log_header)
                tqdm.write(self.log_template.format(epoch + 1, self.iterations, epoch + 1, self.args.epochs,
                                                    dev_acc_coarse, dev_precision_coarse, dev_recall_coarse,
                                                    dev_f1_coarse, dev_loss_coarse))
                tqdm.write('FINE: ' + self.log_header)
                tqdm.write(self.log_template.format(epoch + 1, self.iterations, epoch + 1, self.args.epochs,
                                                    dev_acc_fine, dev_precision_fine, dev_recall_fine,
                                                    dev_f1_fine, dev_loss_fine))

                # Update validation results
                if dev_f1_fine > self.best_dev_f1:
                    self.unimproved_iters = 0
                    self.best_dev_f1 = dev_f1_fine
                    torch.save(self.model_coarse, self.snapshot_path_coarse)
                    torch.save(self.model_fine, self.snapshot_path_fine)

                else:
                    self.unimproved_iters += 1
                    if self.unimproved_iters >= self.args.patience:
                        self.early_stop = True
                        tqdm.write("Early Stopping. Epoch: {}, Best Dev F1: {}".format(epoch, self.best_dev_f1))
                        break
        end_time = time.monotonic()
        print('End training: ', datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        print('Time elapsed: ', end_time-start_time)
