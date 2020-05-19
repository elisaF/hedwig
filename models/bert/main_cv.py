from collections import defaultdict
import csv
import json
import fnmatch
import numpy as np
import os
from models.bert.__main__ import run_main
from models.bert.args import get_args


def get_files_with_pattern(dir_, pattern):
    matching_files = []
    for f in os.listdir(dir_):
        if fnmatch.fnmatch(f, pattern):
            matching_files.append(f)
    return matching_files


def process_json_results(json_prefix, save_file, split, label_suffix=''):
    json_pattern = json_prefix+'*.json_' + split
    json_files = get_files_with_pattern('.', json_pattern)
    results_dict = process_model_results(json_files, label_suffix)
    with open(save_file, 'w') as f:
        header_row = (list(results_dict.values())[0].keys())
        w = csv.writer(f)
        w.writerow(header_row)
        for results in results_dict.items():
            row = results.values()
            w.writerow(row)


def process_model_results(json_files, label_suffix):
    ignore_keys = ['support_class', 'confusion_matrix', 'label_set_info (id/gold/pred)', 'id_gold_pred']
    ignore_keys = [key+label_suffix for key in ignore_keys]

    # read all json files
    result_dicts = []
    for json_file in json_files:
        print('Reading json ', json_file)
        with open(json_file, 'r') as json_data:
            result_dicts.append(json.load(json_data))
    # merge dicts into one
    merged_dict = defaultdict(list)
    for d in result_dicts:
        for key, value in d.items():
            merged_dict[key].append(value)

    # calculate mean and variance
    summarized_dict = {}
    for key, value in merged_dict.items():
        if key not in ignore_keys:
            avg_key = key + '_avg'
            avg_value = np.mean(np.array(value), axis=0)
            summarized_dict[avg_key] = avg_value

            var_key = key + '_var'
            var_value = np.var(np.array(value), axis=0)
            summarized_dict[var_key] = var_value
    return summarized_dict


if __name__ == '__main__':
    args = get_args()

    if args.num_folds < 2:
        raise ValueError("Number of folds must be greater than 1!", args.num_folds)

    orig_metrics_json = args.metrics_json
    for fold in range(0, args.num_folds):
        args.fold_num = fold
        if orig_metrics_json:
            args.metrics_json = orig_metrics_json + '_fold' + str(fold)
        run_main(args)

    # summarize fold results and save to file
    process_json_results(orig_metrics_json, orig_metrics_json+'_summary', 'test')



