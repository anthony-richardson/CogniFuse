from abc import ABC, abstractmethod
import os
import json


class TrainPlatform(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def report_scalar(self, name, value, iteration, group_name=None):
        pass

    @abstractmethod
    def report_args(self, args, name):
        pass

    @abstractmethod
    def close(self):
        pass


class TensorboardPlatform(TrainPlatform):
    def __init__(self, log_dir, args_dir):
        super().__init__()
        self.args_dir = args_dir
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=log_dir)

    def report_scalar(self, name, value, iteration, group_name=None):
        self.writer.add_scalar(f'{group_name}/{name}', value, iteration)

    def report_args(self, args, name):
        if not name.endswith('.json'):
            name += '.json'
        if not os.path.exists(self.args_dir):
            os.makedirs(self.args_dir)
            args_path = os.path.join(self.args_dir, name)
            with open(args_path, 'w') as fw:
                json.dump(vars(args), fw, indent=4)

    def close(self):
        self.writer.close()
