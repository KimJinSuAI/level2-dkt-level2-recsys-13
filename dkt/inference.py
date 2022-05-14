import os

import torch
from args import parse_args
from dkt import trainer
from dkt.dataloader import Preprocess
from dkt.ensemble import average_ensemble, weighted_ensemble


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    preprocess = Preprocess(args)
    preprocess.load_test_data(args.test_file_name)
    test_data = preprocess.get_test_data()

    model_paths = trainer.get_model_paths(args)
    model_paths.sort(key = lambda x: float(x[-7:-3]))

    if args.cv == True:
        all_model_preds = []
        weights = []
        for model_path in model_paths:
            model_preds = trainer.inference(args, test_data, model_path)
            all_model_preds.append(model_preds)
            weights.append(float(model_path[-7:-3]))
        
        total_preds = weighted_ensemble(all_model_preds, weights)
        trainer.write_submission(args, total_preds)
    else:
        model_path = model_paths[-1]
        total_preds =  trainer.inference(args, test_data, model_path)
        trainer.write_submission(args, total_preds)


if __name__ == "__main__":
    args = parse_args(mode="train")
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
