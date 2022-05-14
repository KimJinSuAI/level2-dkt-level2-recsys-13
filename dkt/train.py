import os

import torch
import wandb
from args import parse_args
from dkt import trainer
from dkt.dataloader import Preprocess
from dkt.utils import setSeeds
from sklearn.model_selection import KFold


def main(args):
    if args.wandb:
        wandb.login()
        wandb.init(project="dkt", config=vars(args))

    setSeeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()

    if args.cv == True:
        aucs = []
        kf = KFold(5, shuffle=False)
        for k, (tr_idx, val_idx) in enumerate(kf.split(train_data)):
            tr_data, val_data = train_data[tr_idx], train_data[val_idx]
            aucs.append( trainer.run(args, tr_data, val_data, k) )
        print(sum(aucs)/len(aucs))
    else:
        train_data, valid_data = preprocess.split_data(train_data)     
        trainer.run(args, train_data, valid_data)


if __name__ == "__main__":
    args = parse_args(mode="train")
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
