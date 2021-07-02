import glob
import os

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from datasets.SimpleNumberDataset import SimpleNumberDataset
from datasets.squaredataclass import SquareDataset
from src.set_utils.simple_matcher import SimpleMatcher
from src.set_utils.variance_matcher import VarianceMatcher

torch.set_grad_enabled(True)

"""
To run the logger, enter
    tensorboard --logdir=lightning_logs
"""


class VariancePointNet(pl.LightningModule):
    """
    A naive implementation of pointnet.
    """

    def __init__(self, env_len=1, obj_in_len=2, obj_reg_len=2, obj_attri_len=2, out_set_size=4, hidden_dim=256,
                 num_lstm_layer=1):
        super().__init__()
        self.save_hyperparameters()

        self.hidden_dim = hidden_dim
        self.out_set_size = out_set_size
        self.obj_in_len = obj_in_len
        self.obj_reg_len = obj_reg_len
        self.obj_attri_len = obj_attri_len
        self.env_len = env_len
        self.num_lstm_layer = num_lstm_layer

        self.dropout = nn.Dropout(p=0.1)

        # Embedding layers
        self.obj_embed = nn.Sequential(
            nn.Linear(obj_in_len, 64),
            nn.ReLU(),
        )
        self.obj_encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1024),
            self.dropout,
            nn.ReLU()
        )

        self.env_embed = nn.Sequential(
            nn.Linear(env_len, 64),
            nn.ReLU()
        )

        self.set_embed = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim)),
            nn.ReLU()
        )

        # Output heads

        self.linear_attri = nn.Sequential(
            nn.Linear(1152, 512),
            nn.ReLU(),
            self.dropout,
            nn.Linear(512, 256),
            nn.ReLU(),
            self.dropout,
            nn.Linear(256, 128),
            nn.ReLU(),
            self.dropout,
            nn.Linear(128, obj_attri_len),
            # nn.Sigmoid()
        )
        # self.linear_regs = []
        # for i in range(4):
        #     self.linear_regs.append(nn.Sequential(
        #         nn.Linear(1152, 512),
        #         nn.ReLU(),
        #         self.dropout,
        #         nn.Linear(512, 256),
        #         nn.ReLU(),
        #         self.dropout,
        #         nn.Linear(256, 128),
        #         nn.ReLU(),
        #         self.dropout,
        #         nn.Linear(128, 64),
        #         nn.ReLU(),
        #         self.dropout,
        #         nn.Linear(64, 2 * obj_reg_len),
        #     ))
        self.linear_reg = nn.Sequential(
                nn.Linear(1152, 512),
                nn.ReLU(),
                self.dropout,
                nn.Linear(512, 256),
                nn.ReLU(),
                self.dropout,
                nn.Linear(256, 128),
                nn.ReLU(),
                self.dropout,
                nn.Linear(128, 64),
                nn.ReLU(),
                self.dropout,
                nn.Linear(64, 2 * obj_reg_len),
            )
        self.relu = nn.ReLU()

        # New objects generations
        # self.new_obj_net = nn.LSTM(
        #     input_size=hidden_dim, hidden_size=3 * hidden_dim,
        #     num_layers=self.num_lstm_layer, batch_first=True
        # )

        # Output masks
        self.mask_softmax = nn.Softmax(dim=2)

        # Loss + matching calculation
        self.matcher = VarianceMatcher()
        self.loss_reg_criterion = nn.MSELoss()
        self.loss_mask_criterion = nn.MSELoss()

        self.loss_reg_weight = 1
        self.loss_mask_weight = 0

    def forward(self, objs, env, debug=False):
        """
        Input size: [BATCH_SIZE, N, M]
        """
        # batch size
        bs = env.shape[0]

        # Calculate the object embedding
        # objs = x[:, self.env_len:None].reshape(bs, -1, self.obj_in_len)
        emb_objs = self.obj_embed(objs)                             # Shape: [BS, N, 64]
        h_objs = self.obj_encoder(emb_objs)
        # print(h_objs.shape)                                   # Shape: [BS, N, HIDDEN_DIM]

        # Zero-padding the object embedding
        # pad_size = self.out_set_size - h_objs.shape[1]
        # # pad = nn.ZeroPad2d((0, 0, 0, pad_size))
        # # h_objs = pad(h_objs)                                                      # Shape: [BS, M, HIDDEN_DIM]
        # pad = torch.zeros((bs, pad_size, self.hidden_dim), device=self.device)      # Shape: [BS, M-N, HIDDEN_DIM]
        # h_objs = torch.cat((h_objs, pad), dim=1)  # Shape: [BS, M, HIDDEN_DIM]

        in_set_size = h_objs.shape[1]

        # Calculate the environment embedding
        env = env
        # env = x[:, 0:self.env_len]
        h_env_vector = self.env_embed(env)
        h_env = h_env_vector.repeat((1, in_set_size, 1))  # Shape: [BS, M, 3*HIDDEN_DIM]
        # h_env = h_env_vector.repeat((self.num_lstm_layer, 1, 1))  # Shape: [BS, M, 3*HIDDEN_DIM]

        if debug:
            print("env")
            print(env)
            print(h_env_vector)
        # print(h_env.shape)

        # Obtain the set information by taking the maximum
        h_set_vector, _ = torch.max(h_objs, dim=1)                      # Shape: [BS, HIDDEN_DIM]
        h_set = h_set_vector.repeat((1, in_set_size, 1))                # Shape: [BS, M, 3*HIDDEN_DIM]
        # h_set = h_set_vector.repeat((self.num_lstm_layer, 1, 1))      # Shape: [BS, M, 3*HIDDEN_DIM]

        if debug:
            print("set")
            print(h_set_vector)

        if debug:
            print("objects")
            print(h_objs)

        # Concat the three matrix
        h = torch.cat((emb_objs, h_set, h_env), dim=2)                # Shape: [BS, M, 3*HIDDEN_DIM]
        # h_global = torch.cat((h_env_vector, h_set_vector), dim=1)   # Shape: [BS, 2*HIDDEN_DIM]

        # LSTM version
        # h0 = h_set
        # c0 = h_env
        # h, (_, _) = self.new_obj_net(h_objs, (h0, c0))
        # h = self.decoder(h)

        # Predict
        # pred_reg_result = []
        # for batch in range(bs):
        #     batch_result = []
        #     for i in range(in_set_size):
        #         reg = self.linear_regs[i](h[batch,i,:])
        #         batch_result.append(reg)
        #
        #     batch_result = torch.cat(batch_result, dim=1)
        #     pred_reg_result.append(batch_result)
        #
        # pred_reg_result = self.cat(pred_reg_result, dim=0)

        pred_reg_result = self.linear_reg(h)
        pred_reg_vel = pred_reg_result[:, :, 0:self.obj_reg_len]
        pred_reg = objs[:, :, 0:self.obj_reg_len] + pred_reg_vel
        pred_reg_var = pred_reg_result[:, :, self.obj_reg_len:None]
        pred_reg_var = pred_reg_var**2 + 1e-6           # Prevent zero covariance causing errors
        pred_attri = self.linear_attri(h)

        # Extract the output masks
        pred_mask = pred_attri[:, :, 0]

        # pred_pos = objs + pred_delta
        if debug:
            print("pred_reg")
            print(pred_reg)

        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_attri': pred_attri,
                'pred_mask': pred_mask,
                'pred_reg': pred_reg,
                'pred_reg_var': pred_reg_var}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def loss_fn(self, pred, gt_label):
        # Perform the matching
        match_results = self.matcher(pred, gt_label)

        # Calculate the loss for regression and masking seperately
        loss_reg = 0
        loss_reg_var = 0
        loss_mask = 0

        # Iterate through all batches
        for batch_idx, match_result in enumerate(match_results):
            pred_raw = pred["pred_reg"][batch_idx]
            pred_reg_raw = pred_raw[:, 0:2]
            pred_reg_var_raw = pred["pred_reg_var"][batch_idx]

            gt_raw = gt_label[batch_idx]
            gt_raw_reg = gt_raw[:, 0:2]

            pred_var_reordered = match_result["pred_var_reordered"]
            pred_matched = match_result['pred_reordered']
            gt_matched = match_result['gt_reordered']

            # Loss for regression (position)
            # loss_reg += self.loss_reg_criterion(pred_matched, gt_matched)
            loss_reg += self.loss_reg_criterion(pred_reg_raw, gt_raw_reg)

            # Alternative:
            # print(gt_matched)
            # print(pred_matched)
            # loss_reg += torch.mean((gt_raw - pred_raw)**2 / (2*pred_reg_var_raw) + 0.5*torch.log(pred_reg_var_raw))
            # loss_reg += torch.mean((gt_matched - pred_matched) ** 2)

            # Loss for regression variance
            # pred_diff_square = match_result['pred_diff'] ** 2
            # zero_diff = torch.zeros_like(pred_diff_square, device=self.device)
            # loss_reg_var += self.loss_reg_criterion(pred_diff_square, zero_diff)
            loss_reg_var += torch.mean((torch.abs(pred_reg_var_raw) - torch.abs(gt_raw_reg - pred_reg_raw)) ** 2)

            # Loss for output mask
            pred_mask = pred['pred_mask'][batch_idx]
            tgt_mask = match_result['tgt_mask']
            # print("Pred mask", pred_mask.shape)
            # print("Tgt mask", tgt_mask.shape)
            loss_mask = self.loss_mask_criterion(pred_mask, tgt_mask)

        # summing all the losses
        loss = (loss_reg + loss_reg_var) * self.loss_reg_weight + loss_mask * self.loss_mask_weight

        return loss, loss_reg, loss_reg_var, loss_mask

    def training_step(self, train_batch, batch_idx):
        # Calculate the prediction
        s, a, sprime, sappear, r = train_batch
        pred = self.forward(s, a)

        # Calculate the loss
        loss, loss_reg, loss_reg_var, loss_mask = self.loss_fn(pred, sprime)

        self.log('train_loss', loss)
        self.log('train_reg_loss', loss_reg)
        self.log('train_reg_var_loss', loss_reg_var)
        self.log('train_mask_loss', loss_mask)

        return loss

    def validation_step(self, val_batch, batch_idx):
        # Calculate the prediction
        s, a, sprime, r = val_batch
        pred = self.forward(s, a)

        # Calculate the loss
        loss, loss_reg, loss_reg_var, loss_mask = self.loss_fn(pred, sprime)

        self.log('val_loss', loss)
        self.log('val_reg_loss', loss_reg)
        self.log('val_reg_var_loss', loss_reg_var)
        self.log('val_mask_loss', loss_mask)

    def run_on_batch(self, batch, debug=False):
        # Calculate the prediction
        x, y = batch
        pred = self.forward(x)

        # Calculate the loss
        loss, loss_reg, loss_reg_var, loss_mask = self.loss_fn(pred, y)

        print("Start environment")
        print(x[0, 0:self.env_len])
        print("Start objects")
        print(x[0, self.env_len:None].reshape(-1, self.obj_in_len))

        # Considers the masks
        # pred_reg = pred['pred_reg']
        # pred_mask = pred['pred_mask']
        # pred_mask = (pred_mask[:, :, 1] < pred_mask[:, :, 0])
        # n_pred = torch.sum(pred_mask)
        # print("Prob")
        # print(pred['pred_mask'])

        # print("Number (pred/gt)")
        # print(n_pred.item(), "/", gt_matched.shape[0])

        # View the matched results
        match_results = self.matcher(pred, y)
        for i, match_dict in enumerate(match_results):
            print("Pred_var_reordered")
            print(match_dict["pred_var_reordered"])
            print("Pred_reordered")
            print(match_dict["pred_reordered"])
            print("GT_reordered")
            print(match_dict["gt_reordered"])
            print()


def train_pl():
    """
    Tests
    """
    # Prepare the dataset

    # Square linear
    dataset = SquareDataset(5000, generator_type="linear")
    env_len = 1
    obj_in_len = 2
    obj_reg_len = 2
    obj_attri_len = 2
    out_set_size = 4
    hidden_dim = 384

    # Square rotation
    # dataset = SquareDataset(5000, generator_type="rotation", generate_noise=True)
    # env_len = 1
    # obj_in_len = 2
    # obj_reg_len = 2
    # obj_attri_len = 2
    # out_set_size = 4
    # hidden_dim = 512

    # Simple Number
    # dataset = SimpleNumberDataset(2000, 10, 1, 0.1)
    # env_len=2
    # obj_in_len=1
    # obj_reg_len=1
    # obj_attri_len=2
    # out_set_size=20
    # hidden_dim=32

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
    # dataset = SquareDataset(5, generator_type="linear")
    dataset = SquareDataset(5, generator_type="rotation", generate_noise=False)
    # dataset = SimpleNumberDataset(3, 10, 1, 0.1)
    eval_data_loader = DataLoader(dataset, batch_size=1)
    for batch_idx, batch in enumerate(eval_data_loader):
        model.run_on_batch(batch)


if __name__ == "__main__":
    # train_pl()
    evaluate()
