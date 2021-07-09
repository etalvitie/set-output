from src.dspn.SetDSPN import SetDSPN
from src.simple_pointnet.variance_pointnet import VariancePointNet

import os
import glob

import torch


class PredictionModel:
    def __init__(self,
                 exist_ckpt_path=None,
                 appear_ckpt_path=None,
                 train_exist=True,
                 train_appear=True,
                 obj_in_len=None,
                 env_len=None,
                 obj_reg_len=2,
                 obj_attri_len=2,
                 new_set_size=3):

        # Element-wise prediction model for existing objects
        self.train_exist = train_exist
        if exist_ckpt_path is not None:
            self.exist_model = VariancePointNet.load_from_checkpoint(exist_ckpt_path)
        else:
            self.exist_model = VariancePointNet(
                env_len=env_len,
                obj_in_len=obj_in_len,
                obj_reg_len=obj_reg_len,
                obj_attri_len=obj_attri_len,
                hidden_dim=512
            )

        # Prediction model for appearing objects
        self.train_appear = train_appear
        if appear_ckpt_path is not None:
            self.appear_model = SetDSPN.load_from_checkpoint(appear_ckpt_path)
        else:
            self.appear_model = SetDSPN(
                obj_in_len=obj_in_len,
                obj_reg_len=obj_reg_len,
                obj_attri_len=obj_attri_len,
                env_len=env_len,
                latent_dim=64,
                out_set_size=new_set_size,
                n_iters=10,
                internal_lr=50,
                overall_lr=1e-3,
                loss_encoder_weight=1
            )

        # Decide whether to run the model on CPU or GPU
        dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(dev)
        self.exist_model = self.exist_model.to(self.device)
        self.appear_model = self.appear_model.to(self.device)
        print("Using GPU?", "True" if dev == 'cuda:0' else "False")

        # Optimizers
        self.exist_optimizer = torch.optim.Adam(self.exist_model.parameters(), lr=1e-3)
        self.appear_optimizer = torch.optim.Adam(self.appear_model.parameters(), lr=1e-3)

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
        # Converts input into the format that the existing model needs
        train_batch = [s, a, sprime, sappear, [r]]
        for i, stuff in enumerate(train_batch):
            train_batch[i] = self._tensorfy(stuff)

        # Update existing obj model
        if self.train_exist:
            exist_loss = self.exist_model.training_step(train_batch, self.iter_count)
            self.exist_optimizer.zero_grad()
            exist_loss.backward()
            self.exist_optimizer.step()

        # Update appearing obj model
        if self.train_appear:
            appear_loss = self.appear_model.training_step(train_batch, self.iter_count)
            self.appear_optimizer.zero_grad()
            appear_loss.backward()
            self.appear_optimizer.step()

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
        # Formats the input
        s = self._tensorfy(s)
        a = self._tensorfy(a)

        # Predicts the existing objects and new objects
        sprime = self.exist_model(s, a)['pred_reg']
        sappear = self.appear_model(s, a)['pred_reg']
        
        # Concatenate the two matrixes
        s_ = torch.cat([sprime, sappear], dim=1)

        # Converts back to Python Array
        sprime = sprime.detach().cpu().numpy().tolist()[0]
        sappear = sappear.detach().cpu().numpy().tolist()[0]
        s_ = s_.detach().cpu().numpy().tolist()[0]
        return s_, sprime, sappear

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


"""
Test code
"""
def main():
    model = PredictionModel(obj_in_len=8, env_len=6)

    s = [[0] * 8] * 2
    a = [0] * 6
    sprime = [[0] * 8] * 2
    sappear = [[0] * 8] * 3
    r = 0

    model.updateModel(s, a, sprime, sappear, r)

    return 0

if __name__ == '__main__':
    main()