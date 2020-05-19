from models.bert.__main__ import run_main
from models.bert.args import get_args

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
