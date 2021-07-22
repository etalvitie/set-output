import random

from src.dspn.SetDSPN import SetDSPN
from src.simple_pointnet.variance_pointnet import VariancePointNet
from src.simple_pointnet.num_pointnet import NumPointnet

import os
import glob
from pathlib import Path
from datetime import datetime

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


def load(ckpt_path):
    """ Helper function for loading a model """
    return PredictionModel.load_from_checkpoint(ckpt_path)

class PredictionModel(pl.LightningModule):
    def __init__(self,
                 train_exist=True,
                 train_appear=True,
                 train_rwd=True,
                 obj_in_len=9,
                 env_len=6,
                 obj_reg_len=2,
                 obj_type_len=2,
                 appear_set_size=3,
                 accumulate_batches=64,
                 exist_type_separate=False,
                 appear_type_separate=False,
                 seed=None):

        super().__init__()
        self.save_hyperparameters()

        # Element-wise prediction model for existing objects
        self.train_exist = train_exist
        if seed is not None:
            torch.manual_seed(seed)
        self.exist_model = VariancePointNet(
            env_len=env_len,
            obj_in_len=obj_in_len,
            obj_reg_len=obj_reg_len,
            obj_type_len=obj_type_len,
            hidden_dim=512,
            type_separate=exist_type_separate
        )

        # Prediction model for appearing objects
        self.train_appear = train_appear
        if seed is not None:
            torch.manual_seed(seed)
        self.appear_model = SetDSPN(
            obj_in_len=obj_in_len,
            obj_reg_len=obj_reg_len,
            obj_type_len=obj_type_len,
            env_len=env_len,
            latent_dim=64,
            out_set_size=appear_set_size,
            n_iters=10,
            internal_lr=50,
            overall_lr=1e-3,
            loss_encoder_weight=1,
            type_separate=appear_type_separate
        )

        # Prediction model for rewards
        self.train_rwd = train_rwd
        if seed is not None:
            torch.manual_seed(seed)
        self.rwd_model = NumPointnet(
            env_len=env_len,
            obj_in_len=obj_in_len,
            out_len=1,
            variance_type="separate"
        )

        # if ckpt_path is not None:
        #     checkpoint = torch.load(ckpt_path)
        #     self.load_state_dict(checkpoint['model_state_dict'])

        # Decide whether to run the model on CPU or GPU
        # dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # self.device = torch.device(dev)
        # self.exist_model = self.exist_model.to(self.device)
        # self.appear_model = self.appear_model.to(self.device)
        # self.rwd_model = self.rwd_model.to(self.device)
        # print("Using GPU?", "True" if dev == 'cuda:0' else "False")

        # Optimizers
        self.exist_optimizer = torch.optim.Adam(self.exist_model.parameters(), lr=1e-3)
        self.appear_optimizer = torch.optim.Adam(self.appear_model.parameters(), lr=1e-3)
        self.rwd_optimizer = torch.optim.Adam(self.rwd_model.parameters(), lr=1e-3)

        # Trainer for size calculation and saving
        empty_loader = DataLoader(None)
        trainer = pl.Trainer(default_root_dir='ckpts',
                                  max_steps=0,
                                  num_sanity_val_steps=0)
        trainer.fit(self, empty_loader)

        # Accumulate grad batches
        self.accumulate_batches = accumulate_batches
        self.exist_loss, self.appear_loss, self.rwd_loss = 0, 0, 0

        # Flags and logging
        self.iter_count = 0

    def updateModel(self, s, a, sprime, sappear, r):
        """
        Update the prediction model using labeled data
        Args:
            s: [Python Array] current object states.
            a: [Python Array] action taken.
            sprime: [Python Array] of next frame object states of existing objects.
            sappear: [Python Array] of next frame object states of appearing objects.
            r:  [Float] Reward of the action state pair.
        """
        # Set to train mode
        self.train()

        # Converts input into the format that the existing model needs
        train_batch = [s, a, sprime, sappear, [r]]
        for i, stuff in enumerate(train_batch):
            train_batch[i] = self._tensorfy(stuff)

        # Accumulates the losses
        self.exist_loss += self.exist_model.training_step(train_batch, self.iter_count)
        self.appear_loss += self.appear_model.training_step(train_batch, self.iter_count)
        self.rwd_loss += self.rwd_model.training_step(train_batch, self.iter_count)

        # Update the models when there are enough batches
        if self.iter_count % self.accumulate_batches == 0:
            self.exist_loss /= self.accumulate_batches
            self.appear_loss /= self.accumulate_batches
            self.rwd_loss /= self.accumulate_batches

            # Update existing obj model
            torch.autograd.set_detect_anomaly(True)
            if self.train_exist:
                self.exist_optimizer.zero_grad()
                self.exist_loss.backward(retain_graph=True)
                self.exist_optimizer.step()

            # Update appearing obj model
            if self.train_appear:
                self.appear_optimizer.zero_grad()
                self.appear_loss.backward(retain_graph=True)
                self.appear_optimizer.step()

            # Update reward model
            if self.train_rwd:
                self.rwd_optimizer.zero_grad()
                self.rwd_loss.backward()
                self.rwd_optimizer.step()

            # Reset loss
            self.exist_loss = 0
            self.appear_loss = 0
            self.rwd_loss = 0

        # Increase iter counter
        self.iter_count += 1

        return

    def predict(self, s, a):
        """
        Makes prediction on the object states in the next frame based the current object states and player action.
        Args:
            s: [torch.Tensor] Current object states of size [1xMxN]
            a: [torch.Tensor] Action vector of size [1xV]
        Returns:
            [s_, sprime, sappear]
            s_: Prediction set: sprime + sappear
            sprime: The set of existing objects
            sappear: The set of appearing objects
        """
        # Set to evaluation mode
        self.eval()

        # Formats the input
        s = self._tensorfy(s)
        a = self._tensorfy(a)

        # Predicts the existing objects and new objects
        existResults = self.exist_model(s, a)
        appearResults = self.appear_model(s, a)

        sprime = existResults['pred_reg']
        visprime = existResults['pred_mask']
        sappear = appearResults['pred_reg']
        visappear = appearResults['pred_mask']
        rwd = self.rwd_model(s, a)['pred_val']
        
        # Concatenate the two matrixes
        # s_ = torch.cat([sprime, sappear], dim=1)
        # s_ = s_.detach().cpu().numpy().tolist()[0]

        # Converts back to Python Array
        sprime = sprime.detach().cpu().numpy().tolist()[0]
        sappear = sappear.detach().cpu().numpy().tolist()[0]
        rwd = rwd.detach().cpu().numpy().tolist()[0][0]
        visprime = visprime.detach().cpu().numpy().tolist()[0]
        visappear = visappear.detach().cpu().numpy().tolist()[0]

        sprime = [i + [j] for i, j in zip(sprime, visprime)]
        sappear = [i + [j] for i, j in zip(sappear, visappear)]

        s_ = sprime + sappear

        return s_, sprime, sappear, rwd 

    def save(self, path=None):
        # Create the checkpoint folder if not existed
        Path("ckpts").mkdir(parents=True, exist_ok=True)

        # Determine the right path
        if path is None:
            now = datetime.now()
            date_time = now.strftime("%m_%d_%H_%M")
            filename = date_time + '.ckpt'
            path = os.path.join("ckpts", filename)

        # Save the model
        # torch.save({
        #     'iter': self.iter_count,
        #     'model_state_dict': self.state_dict(),
        #     # 'exits_model_state_dict': self.exist_model.state_dict(),
        #     # 'appear_model_state_dict': self.appear_model.state_dict(),
        #     # 'rwd_model_state_dict': self.rwd_model.state_dict(),
        #     'exist_optimizer_state_dict': self.exist_optimizer.state_dict(),
        #     'appear_optimizer_state_dict': self.appear_optimizer.state_dict(),
        #     'rwd_optimizer_state_dict': self.rwd_optimizer.state_dict()
        # }, path)

        # Create an empty dataloader to make pl happy
        empty_loader = DataLoader(None)
        trainer = pl.Trainer(default_root_dir='ckpts',
                             max_steps=0,
                             num_sanity_val_steps=0)
        trainer.fit(self, empty_loader)
        trainer.save_checkpoint(path)


    def _tensorfy(self, m):
        """
        Notice: assuming that the input is provided as Python list.
        Converts the input vector/matrix to the desired format of the model (torch Tensor)
        Args:
            m: input vector/matrix
        Returns:
            m_: formatted torch Tensor
        """

        # Converts into Tensor
        m_ = torch.Tensor(m)
        m_ = m_.to(self.device)

        # Prepares it in the batch format
        m_ = m_.unsqueeze(0)

        return m_

    def training_step(self, *args, **kwargs):
        pass

    def train_dataloader(self):
        pass

    def configure_optimizers(self):
        pass


def main():
    """
    Test code
    """
    # Train dataset
    from datasets.MinatarDataset.MinatarDataset import MinatarDataset
    dataset = MinatarDataset(name="asterix_dataset_random_3000.json")
    dim_dict = dataset.get_dims()
    env_len = dim_dict["action_len"]
    obj_in_len = dim_dict["obj_len"]
    type_len = dim_dict["type_len"]

    # Constrcut the model
    model = PredictionModel(obj_in_len=obj_in_len,
                            env_len=env_len,
                            obj_type_len=type_len,
                            accumulate_batches=4,
                            exist_type_separate=True,
                            appear_type_separate=True)
    # model = load(ckpt_path="ckpts/")

    # Train
    for _ in range(5):
        idx = random.randint(0, len(dataset))
        batch = dataset[idx] # s, a, sprime, sappear, r
        batch_ = []
        for item in batch:
            batch_.append(item.numpy().tolist())
        batch_[-1] = batch_[-1][0]
        model.updateModel(*batch_)

    model.save()

    return 0

if __name__ == '__main__':
    main()