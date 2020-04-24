import random
import json

import numpy as np
import torch
from transformers import AdamW, BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup

from common.constants import *
from common.evaluators.bert_evaluator import BertEvaluator
from common.trainers.bert_hierarchical_trainer import BertHierarchicalTrainer
from datasets.bert_processors.aapd_processor import AAPDProcessor
from datasets.bert_processors.agnews_processor import AGNewsProcessor
from datasets.bert_processors.imdb_processor import IMDBProcessor
from datasets.bert_processors.reuters_processor import ReutersProcessor
from datasets.bert_processors.congressional_hearing_processor import CongressionalHearingProcessor
from datasets.bert_processors.sogou_processor import SogouProcessor
from datasets.bert_processors.sst_processor import SST2Processor
from datasets.bert_processors.yelp2014_processor import Yelp2014Processor
from models.bert.args import get_args


def evaluate_split(model, processor, tokenizer, args, save_file, split='dev', is_coarse=False):
    evaluator = BertEvaluator(model, processor, tokenizer, args, split, is_coarse)
    scores, score_names = evaluator.get_scores(silent=True)
    accuracy, precision, recall, f1, avg_loss = scores[:5]
    if is_coarse:
        print('\n' + 'FINE: ' + LOG_HEADER)
    else:
        print('\n' + 'COARSE: ' + LOG_HEADER)
    print(LOG_TEMPLATE.format(split.upper(), accuracy, precision, recall, f1, avg_loss))

    scores_dict = dict(zip(score_names, scores))
    with open(save_file, 'w') as f:
        f.write(json.dumps(scores_dict))


if __name__ == '__main__':
    # Set default configuration in args.py
    args = get_args()
    print('Args: ', args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    print('Device:', str(device).upper())
    print('Number of GPUs:', n_gpu)
    print('FP16:', args.fp16)

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    metrics_dev_json_coarse = args.metrics_json + '_dev_coarse'
    metrics_dev_json_fine = args.metrics_json + '_dev_fine'
    metrics_test_json_coarse = args.metrics_json + '_test_coarse'
    metrics_test_json_fine = args.metrics_json + '_test_fine'

    dataset_map = {
        'SST-2': SST2Processor,
        'Reuters': ReutersProcessor,
        'CongressionalHearing': CongressionalHearingProcessor,
        'IMDB': IMDBProcessor,
        'AAPD': AAPDProcessor,
        'AGNews': AGNewsProcessor,
        'Yelp2014': Yelp2014Processor,
        'Sogou': SogouProcessor
    }

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    if args.dataset not in dataset_map:
        raise ValueError('Unrecognized dataset')

    args.batch_size = args.batch_size // args.gradient_accumulation_steps
    args.device = device
    args.n_gpu = n_gpu
    args.num_labels = dataset_map[args.dataset].NUM_CLASSES
    args.is_multilabel = dataset_map[args.dataset].IS_MULTILABEL
    args.parent_to_child_index_map = {0: (0, 1), 1: (2, 3), 2: (4, 5)}

    if not args.trained_model:
        save_path = os.path.join(args.save_path, dataset_map[args.dataset].NAME+'_coarse')
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(args.save_path, dataset_map[args.dataset].NAME + '_fine')
        os.makedirs(save_path, exist_ok=True)

    args.is_hierarchical = False
    processor = dataset_map[args.dataset]()
    pretrained_vocab_path = args.model
    tokenizer = BertTokenizer.from_pretrained(pretrained_vocab_path)

    train_examples = None
    num_train_optimization_steps = None
    if not args.trained_model:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.batch_size / args.gradient_accumulation_steps) * args.epochs

    #pretrained_model_path = args.model if os.path.isfile(args.model) else PRETRAINED_MODEL_ARCHIVE_MAP[args.model]
    pretrained_model_path = args.model
    model_coarse = BertForSequenceClassification.from_pretrained(pretrained_model_path, num_labels=args.num_coarse_labels)
    model_fine = BertForSequenceClassification.from_pretrained(pretrained_model_path, num_labels=args.num_labels)
    model_coarse.to(device)
    model_fine.to(device)

    # Prepare optimizer
    param_optimizer = list(model_coarse.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=num_train_optimization_steps,
                                                num_warmup_steps=args.warmup_proportion * num_train_optimization_steps)

    trainer = BertHierarchicalTrainer(model_coarse, model_fine, optimizer, processor, scheduler, tokenizer, args)

    trainer.train()
    model_coarse = torch.load(trainer.snapshot_path_coarse)
    model_fine = torch.load(trainer.snapshot_path_fine)

    if args.evaluate_dev:
        evaluate_split(model_coarse, processor, tokenizer, args, metrics_dev_json_coarse, split='dev', is_coarse=True)
        evaluate_split(model_fine, processor, tokenizer, args, metrics_dev_json_fine, split='dev')
    if args.evaluate_test:
        evaluate_split(model_coarse, processor, tokenizer, args, metrics_test_json_coarse, split='test')
        evaluate_split(model_fine, processor, tokenizer, args, metrics_test_json_fine, split='test')

