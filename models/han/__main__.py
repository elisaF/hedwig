import json
import logging
import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.onnx

from common.evaluate import EvaluatorFactory
from common.train import TrainerFactory
from datasets.aapd import AAPDHierarchical as AAPD
from datasets.imdb import IMDBHierarchical as IMDB
from datasets.reuters import ReutersHierarchical as Reuters
from datasets.congressional_hearing import CongressionalHearingHierarchical as CongressionalHearing
from datasets.yelp2014 import Yelp2014Hierarchical as Yelp2014
from models.han.args import get_args
from models.han.model import HAN


class UnknownWordVecCache(object):
    """
    Caches the first randomly generated word vector for a certain size to make it is reused.
    """
    cache = {}

    @classmethod
    def unk(cls, tensor):
        size_tup = tuple(tensor.size())
        if size_tup not in cls.cache:
            cls.cache[size_tup] = torch.Tensor(tensor.size())
            cls.cache[size_tup].uniform_(-0.25, 0.25)
        return cls.cache[size_tup]


def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def evaluate_dataset(split_name, dataset_cls, model, embedding, loader, batch_size, device, is_multilabel, save_file):
    saved_model_evaluator = EvaluatorFactory.get_evaluator(dataset_cls, model, embedding, loader, batch_size, device)
    if hasattr(saved_model_evaluator, 'is_multilabel'):
        saved_model_evaluator.is_multilabel = is_multilabel
    if hasattr(saved_model_evaluator, 'ignore_lengths'):
        saved_model_evaluator.ignore_lengths = True

    scores, score_names = saved_model_evaluator.get_scores()
    print('Evaluation metrics for', split_name)
    print(score_names)
    print(scores)

    scores_dict = dict(zip(score_names, scores))
    with open(save_file, 'w') as f:
        f.write(json.dumps(scores_dict))


if __name__ == '__main__':
    # Set default configuration in args.py
    args = get_args()
    print('Args: ', args)
    logger = get_logger()

    metrics_dev_json = args.metrics_json + '_dev'
    metrics_test_json = args.metrics_json + '_test'

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if not args.cuda:
        args.gpu = -1
    if torch.cuda.is_available() and args.cuda:
        print('Note: You are using GPU for training')
        torch.cuda.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.cuda:
        print('Warning: Using CPU for training')

    dataset_map = {
        'Reuters': Reuters,
        'CongressionalHearing': CongressionalHearing,
        'AAPD': AAPD,
        'IMDB': IMDB,
        'Yelp2014': Yelp2014
    }

    if args.dataset not in dataset_map:
        raise ValueError('Unrecognized dataset')

    else:
        dataset_class = dataset_map[args.dataset]
        if args.evaluate_dev:
            train_iter, dev_iter = dataset_class.iters_dev(args.data_dir,
                                                           args.word_vectors_file,
                                                           args.word_vectors_dir,
                                                           batch_size=args.batch_size,
                                                           device=device,
                                                           unk_init=UnknownWordVecCache.unk)
        if args.evaluate_test:
            train_iter, test_iter = dataset_class.iters_test(args.data_dir,
                                                             args.word_vectors_file,
                                                             args.word_vectors_dir,
                                                             batch_size=args.batch_size,
                                                             device=device,
                                                             unk_init=UnknownWordVecCache.unk)
    config = deepcopy(args)
    config.dataset = train_iter.dataset
    config.target_class = train_iter.dataset.NUM_CLASSES
    config.words_num = len(train_iter.dataset.TEXT_FIELD.vocab)

    print('Dataset:', args.dataset)
    print('No. of target classes:', train_iter.dataset.NUM_CLASSES)
    print('No. of train instances', len(train_iter.dataset))
    if args.evaluate_dev:
        print('No. of dev instances', len(dev_iter.dataset))
    if args.evaluate_test:
        print('No. of test instances', len(test_iter.dataset))

    if args.resume_snapshot:
        if args.cuda:
            model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))
        else:
            model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage)
    else:
        model = HAN(config)
        model.to(device)

    if not args.trained_model:
        save_path = os.path.join(args.save_path, dataset_map[args.dataset].NAME)
        os.makedirs(save_path, exist_ok=True)

    parameter = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameter, lr=args.lr, weight_decay=args.weight_decay)
    
    train_evaluator = EvaluatorFactory.get_evaluator(dataset_class, model, None, train_iter, args.batch_size, device)
    if args.evaluate_dev:
        dev_evaluator = EvaluatorFactory.get_evaluator(dataset_class, model, None, dev_iter, args.batch_size, device)
        if hasattr(dev_evaluator, 'is_multilabel'):
            dev_evaluator.is_multilabel = dataset_class.IS_MULTILABEL
        if hasattr(dev_evaluator, 'ignore_lengths'):
            dev_evaluator.ignore_lengths = True

    if args.evaluate_test:
        test_evaluator = EvaluatorFactory.get_evaluator(dataset_class, model, None, test_iter, args.batch_size,
                                                        device)
        if hasattr(test_evaluator, 'is_multilabel'):
            test_evaluator.is_multilabel = dataset_class.IS_MULTILABEL
        if hasattr(test_evaluator, 'ignore_lengths'):
            test_evaluator.ignore_lengths = True

    if hasattr(train_evaluator, 'is_multilabel'):
        train_evaluator.is_multilabel = dataset_class.IS_MULTILABEL

    trainer_config = {
        'optimizer': optimizer,
        'batch_size': args.batch_size,
        'log_interval': args.log_every,
        'patience': args.patience,
        'model_outfile': args.save_path,
        'logger': logger,
        'is_multilabel': dataset_class.IS_MULTILABEL,
        'ignore_lengths': True
    }
    if args.evaluate_dev:
        trainer = TrainerFactory.get_trainer_dev(args.dataset, model, None, train_iter, trainer_config, train_evaluator, dev_evaluator)
    if args.evaluate_test:
        trainer = TrainerFactory.get_trainer_test(args.dataset, model, None, train_iter, trainer_config, train_evaluator,
                                                  test_evaluator)

    if not args.trained_model:
        trainer.train(args.epochs)
    else:
        if args.cuda:
            model = torch.load(args.trained_model, map_location=lambda storage, location: storage.cuda(device))
        else:
            model = torch.load(args.trained_model, map_location=lambda storage, location: storage)

    # Calculate dev and test metrics
    if hasattr(trainer, 'snapshot_path'):
        model = torch.load(trainer.snapshot_path)

    if args.evaluate_dev:
        evaluate_dataset('dev', dataset_map[args.dataset], model, None, dev_iter, args.batch_size,
                         is_multilabel=dataset_class.IS_MULTILABEL,
                         device=device, save_file=metrics_dev_json)
    if args.evaluate_test:
        evaluate_dataset('test', dataset_map[args.dataset], model, None, test_iter, args.batch_size,
                         is_multilabel=dataset_class.IS_MULTILABEL,
                         device=device, save_file=metrics_test_json)
