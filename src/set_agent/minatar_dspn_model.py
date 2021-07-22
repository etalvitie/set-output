"""
Predicting Objects from Minatar dataset
"""
import random

from src.simple_pointnet.variance_pointnet import VariancePointNet
from src.dspn.SetDSPN import SetDSPN
from src.set_transformer import SetTransformer
from datasets.MinatarDataset.MinatarDataset import MinatarDataset

import glob
import os

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
sns.set_style("whitegrid")

def main():
    train_pl()
    # evaluate(path="lightning_logs/version_37/checkpoints/epoch=11-step=5603.ckpt")
    # evaluate(path="lightning_logs/version_36/checkpoints/epoch=9-step=4669.ckpt")


def train_pl():
    # Square linear
    # dataset = MinatarDataset(name="dataset_random_3000_bullet_matched.json")
    # dataset = MinatarDataset(name="dataset_random_3000_new_matched.json")
    # dataset = MinatarDataset(name="dataset_random_3000_full_matched.json")
    dataset = MinatarDataset(name="asterix_dataset_random_3000.json")
    dim_dict = dataset.get_dims()
    env_len = dim_dict["action_len"]
    obj_in_len = dim_dict["obj_len"]
    type_len = dim_dict["type_len"]

    # Prepare the dataloader
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, dataset_size - train_size])
    train_data_loader = DataLoader(train_set, batch_size=1, shuffle=True)  # num_workers=8, pin_memory=True,
    val_data_loader = DataLoader(val_set, batch_size=1, pin_memory=True)

    # Initialize the model
    # model = SetDSPN(
    #     obj_in_len=obj_in_len,
    #     obj_reg_len=2,
    #     obj_type_len=type_len,
    #     env_len=env_len,
    #     latent_dim=64,
    #     out_set_size=3,
    #     n_iters=10,
    #     internal_lr=50,
    #     overall_lr=1e-3,
    #     loss_encoder_weight=1
    # )

    model = SetTransformer(
        obj_in_len=obj_in_len,
        obj_reg_len=2,
        obj_type_len=type_len,
        env_len=env_len,
        out_set_size=3,
        learning_rate=1e-4
    )

    # Early stop callback
    # early_stop_callback = EarlyStopping(
    #     monitor='val_loss',
    #     min_delta=0.00,
    #     patience=3,
    #     verbose=False,
    #     mode='min'
    # )

    # Native train
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # for i, batch in enumerate(train_data_loader):
    #     print(i)
    #     s, a, sprime, sappear, r = batch
    #     s, a, sappear = s.to(model.device), a.to(model.device), sappear.to(model.device)
    #     pred = model(s, a)
    #     losses = model.loss_fn(pred, sappear)
    #
    #     optimizer.zero_grad()
    #     losses['loss_encoder'].backward()
    #     optimizer.step()
    #     pass


    # Train
    gpus = torch.cuda.device_count()
    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        max_epochs=16,
        # check_val_every_n_epoch=4,
        accumulate_grad_batches=64,
        profiler="simple",
        auto_lr_find=True,
        log_every_n_steps=5,
        # callbacks=[early_stop_callback]
    )

    lr_finder = False
    if lr_finder:
        # Find the ideal lr
        lr_finder = trainer.tuner.lr_find(model,
                                          train_dataloader=train_data_loader,
                                          val_dataloaders=val_data_loader,
                                          max_lr=0.1,
                                          min_lr=1e-5)
        # Results can be found in
        lr_finder.results

        # Plot with
        fig = lr_finder.plot(suggest=True)
        fig.show()

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()
    else:
        trainer.fit(model, train_data_loader, val_data_loader)

        # Evaluate
        # trainer.test(model, test_dataloaders = val_data_loader)
        evaluate(model=model)


def evaluate(model=None, path=None):
    print("entered function...")

    # load model
    if model is None:
        if path is None:
            list_ckpts = glob.glob(os.path.join("lightning_logs", "*", "checkpoints", "*.ckpt"))
            latest_ckpt = max(list_ckpts, key=os.path.getctime)
            print("Using checkpoint ", latest_ckpt)
            path = latest_ckpt

        model = SetDSPN.load_from_checkpoint(path)
        # model.freeze()

    print("imported model...")

    # Evaluate
    # dataset = MinatarDataset(name="dataset_random_3000_bullet_matched.json")
    # dataset = MinatarDataset(name="dataset_random_3000_new_matched.json")
    # dataset = MinatarDataset(name="dataset_random_3000_full_matched.json")
    dataset = MinatarDataset(name="asterix_dataset_random_3000_matched.json")
    eval_data_loader = DataLoader(dataset, batch_size=1)

    print("loaded dataset...")

    counter = 0
    while counter < 20:
        batch_idx = random.randint(0, len(dataset))
        batch = dataset[batch_idx]
        s, a, sprime, sappear, r = batch
        if len(sappear) == 0:
            continue

        pred = model(s.unsqueeze(0), a.unsqueeze(0))
        visualize(pred, s, sprime, sappear)
        counter += 1
    
    print("finished...")


def visualize(pred, s, gt_sprime, gt_sappear):
    """ 
        pred -> dictionary from model output
        s -> set input of current state
        gt_sprime -> ground truth set output of existing object
        gt_sappear -> variable size set output of new object
    """
    # Extract the information
    pred_mask = pred['pred_mask'][0].detach()
    pred_pos = pred['pred_reg'][0][:, 0:2].detach()
    pred_pos_var = pred['pred_reg_var'][0][:, 0:2].detach()

    # Plot predictions
    pred_data = {
        "x": pred_pos[:, 0],
        "y": pred_pos[:, 1],
        "var": pred_pos_var[:, 0] + pred_pos_var[:, 1],
        "vis": pred_mask.cpu().numpy(),
        "xvar": pred_pos_var[:, 0],
        "yvar": pred_pos_var[:, 1]
    }

    sns.relplot(
        data=pred_data, x='x', y='y',
        alpha=1, hue='vis',
        legend='full', palette='flare'
    )
    plt.errorbar(pred_data['x'], pred_data['y'],
                 xerr=pred_data['xvar'],
                 yerr=pred_data['yvar'],
                 marker='.', c='C3', alpha=0.2,
                 linestyle='None')


    # Plot ground truth
    if len(gt_sappear) != 0:
        gt_data = {
            "x": gt_sappear[:, 0],
            "y": gt_sappear[:, 1]
        }
        plt.plot(gt_sappear[:, 0], gt_sappear[:, 1], 'rx')
        print(gt_sappear)
    plt.plot(gt_sprime[:, 0], gt_sprime[:, 1], 'kx')
    plt.plot(s[:, 0], s[:, 1], 'ko')

    plt.show()


if __name__ == "__main__":
    main()
