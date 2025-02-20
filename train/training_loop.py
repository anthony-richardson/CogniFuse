import blobfile as bf
import torch
import torch.nn.functional as F
from torch.optim import AdamW, Adam
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm

from utils import logger


class TrainingLoop:
    def __init__(self, args, train_platform, model, task_tools, train_data,
                 validation_data, save_dir):
        self.train_platform = train_platform
        self.model = model
        self.train_data = train_data
        self.validation_data = validation_data
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.save_interval = args.save_interval
        self.weight_decay = args.weight_decay
        self.optimizer = args.optimizer
        self.step = 0
        self.epoch = 0
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.train_data) + 1
        print(f"Number of epochs: {self.num_epochs}")
        self.save_dir = save_dir
        self.f1_score_variant = args.f1_score_variant

        self.task_tools = task_tools

        try:
            self.modality_name = args.modality
        except AttributeError:
            self.modality_name = None

        self.is_multimodal = args.multimodal

        if self.optimizer == 'AdamW':
            self.opt = AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        elif self.optimizer == 'Adam':
            self.opt = Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        else: 
            raise Exception('Unknown optimizer.')

        if args.cuda:
            self.device = torch.device(f"cuda:{args.device}")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)

    def run_loop(self):
        if self.validation_data is not None:
            self.run_validation()

        for epoch in range(self.num_epochs):
            self.epoch = epoch
            train_epoch_info = self.run_epoch(split='train')
            self.report_metrics(train_epoch_info, split='train')

            if (epoch + 1) % self.save_interval == 0:
                if self.validation_data is not None:
                    self.run_validation()
                self.save()

        # Save the last checkpoint if it wasn't already saved.
        if self.num_epochs % self.save_interval != 0:
            if self.validation_data is not None:
                self.run_validation()
            self.save()

    def run_validation(self):
        self.model.eval()
        # Added this to avoid memory overload
        with torch.no_grad():
            validation_epoch_info = self.run_epoch(split='validation')
        self.model.train()
        self.report_metrics(validation_epoch_info, split='validation')

    def run_epoch(self, split):
        if split == 'train':
            data = self.train_data
            desc = f'Epoch {self.epoch}'
        elif split == 'validation':
            data = self.validation_data
            desc = 'Validation round'
        else:
            raise ValueError(f'Unknown split: {split}')

        info = {
            'cross_entropy_per_sample': [],
            'predictions': [],
            'targets': []
        }
        for modality_data, meta_info in tqdm(data, desc=desc):
            if self.is_multimodal:
                x = [modality_data[modality_name].type(torch.float).to(self.device) for
                     modality_name in ['eeg', 'ppg', 'eda', 'resp']]
            else:
                x = modality_data[self.modality_name].type(torch.float).to(self.device)
            # Cross entropy is only implemented on LongTensor
            y_target = self.task_tools.map_meta_info_to_class(
                self.task_tools, meta_info=meta_info).type(torch.LongTensor).to(self.device)

            step_info = self.run_step(x, y_target, split)
            for key in step_info:
                info[key].append(step_info[key])

            if split == 'train':
                self.step += 1

        epoch_info = {key: torch.cat(value, dim=0) for key, value in info.items()}
        return epoch_info

    def run_step(self, x, y_target, split):
        loss, step_info = self.forward(x, y_target)
        if split == 'train':
            loss.backward()
            self.opt.step()
            self.log_step()
        return step_info

    def forward(self, x, y_target):
        self.opt.zero_grad()

        output = self.model(x)
        y_prediction = F.softmax(output, dim=-1)
        predicted_labels = torch.argmax(y_prediction, dim=-1)

        loss = F.cross_entropy(y_prediction, y_target)
        loss_per_sample = F.cross_entropy(y_prediction, y_target, reduction='none')

        step_info = {
            'cross_entropy_per_sample': loss_per_sample,
            'predictions': predicted_labels,
            'targets': y_target
        }
        return loss, step_info

    def report_metrics(self, epoch_info, split):
        metrics = self.calc_metrics(epoch_info)
        log_metrics_dict(metrics, split)

        for k, v in logger.get_current().name2val.items():
            if k in ['step', 'samples'] or '_q' in k or split not in k:
                continue
            else:
                self.train_platform.report_scalar(name=k, value=v, iteration=self.step, group_name='Loss')

    def calc_metrics(self, epoch_info):
        cross_entropy_per_sample = epoch_info['cross_entropy_per_sample']
        predictions = epoch_info['predictions']
        targets = epoch_info['targets']

        cross_entropy = np.mean(cross_entropy_per_sample.cpu().detach().numpy())

        accuracy = accuracy_score(
            targets.cpu().detach().numpy(),
            predictions.cpu().detach().numpy()
        )

        f1 = f1_score(
            targets.cpu().detach().numpy(),
            predictions.cpu().detach().numpy(),
            # When all predictions and labels are negative
            zero_division=1.0,
            average=self.f1_score_variant
        )
        metrics = {
            'cross_entropy': cross_entropy,
            'accuracy': accuracy,
            f'{self.f1_score_variant}_f1_score': f1
        }
        return metrics

    def log_step(self):
        logger.logkv("step", self.step)
        logger.logkv("samples", (self.step + 1) * self.batch_size)

    def ckpt_file_name(self):
        return f"model{self.step:09d}.pt"

    def save(self):
        def save_checkpoint(state_dict):
            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.model.state_dict())

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{self.step:09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)


def get_blob_logdir():
    # Can be changed to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def log_metrics_dict(metrics, split):
    for key, values in metrics.items():
        logger.logkv(f'{split}_{key}', values)
