"""
Predicting Objects from Minatar dataset
"""
import random

from src.simple_pointnet.variance_pointnet import VariancePointNet
from datasets.MinatarDataset.MinatarDataset import MinatarDataset

import glob
import os

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
torch.set_grad_enabled(True)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

def main():
    # train_pl()
    evaluate()


def train_pl():
    # Square linear
    dataset = MinatarDataset(dataset_size=20000)
    dim_dict = dataset.get_dims()
    env_len = dim_dict["action_len"]
    obj_in_len = dim_dict["obj_len"]
    obj_reg_len = 2
    obj_attri_len = 2
    out_set_size = 10
    hidden_dim = 512

    # Prepare the dataloader
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, dataset_size - train_size])
    train_data_loader = DataLoader(train_set, batch_size=1, shuffle=True)  # num_workers=8, pin_memory=True,
    val_data_loader = DataLoader(val_set, batch_size=1, pin_memory=True)

    # Initialize the model
    model = VariancePointNet(
        env_len=env_len,
        obj_in_len=obj_in_len,
        obj_reg_len=obj_reg_len,
        obj_attri_len=obj_attri_len,
        out_set_size=out_set_size,
        hidden_dim=hidden_dim
    )

    # Early stop callback
    # early_stop_callback = EarlyStopping(
    #     monitor='val_loss',
    #     min_delta=0.00,
    #     patience=3,
    #     verbose=False,
    #     mode='min'
    # )

    # Train
    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        max_epochs=4,
        # check_val_every_n_epoch=4,
        accumulate_grad_batches=16,
        profiler="simple"
        # callbacks=[early_stop_callback]
    )
    trainer.fit(model, train_data_loader, val_data_loader)

    # Evaluate
    # trainer.test(model, test_dataloaders = val_data_loader)
    evaluate(model=model)


def evaluate(model=None, path=None):
    # load model
    if model is None:
        if path is None:
            list_ckpts = glob.glob(os.path.join("lightning_logs", "*", "checkpoints", "*.ckpt"))
            latest_ckpt = max(list_ckpts, key=os.path.getctime)
            print("Using checkpoint ", latest_ckpt)
            path = latest_ckpt

        model = VariancePointNet.load_from_checkpoint(path)
        model.freeze()

    # Evaluate
    dataset = MinatarDataset()
    eval_data_loader = DataLoader(dataset, batch_size=1)
    for i in range(5):
        batch_idx = random.randint(0, len(dataset))
        batch = dataset[batch_idx]
        s, a, sprime, r = batch
        pred = model(s.unsqueeze(0), a.unsqueeze(0))
        visualize(pred, sprime, s)


def visualize(pred, gt, s):
    # Extract the information
    pred_mask = pred['pred_mask'][0]
    pred_pos = pred['pred_reg'][0][:, 0:2]
    pred_pos_var = pred['pred_reg_var'][0][:, 0:2]
    pred_pos_var = pred_pos_var[:, 0] + pred_pos_var[:, 1]

    pred_data = {
        "x": pred_pos[:, 0],
        "y": pred_pos[:, 1],
        "var": pred_pos_var
    }
    gt_data = {
        "x": gt[:, 0],
        "y": gt[:, 1]
    }

    # Plot
    sns.relplot(
        data=pred_data, x='x', y='y',
        size='var', alpha=0.5, legend=False
    )
    plt.plot(gt[:, 0], gt[:, 1], 'kx')
    plt.plot(s[:, 0], s[:, 1], 'ko')
    plt.show()


if __name__ == "__main__":
    main()
