from models.bert.__main__ import run_main
from models.bert.args import get_args

if __name__ == '__main__':
    args = get_args()

    if args.num_folds < 2:
        raise ValueError("Number of folds must be greater than 1!", args.num_folds)

    for fold in range(0, args.num_folds):
        args.fold_num = fold
        if args.metrics_json:
            args.metrics_json = args.metrics_json + '_' + str(fold)
        run_main(args)
