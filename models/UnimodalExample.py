# Example of a unimodal model for the benchmark. 

import torch

from utils.model_util import BaseBenchmarkModel


class UnimodalExample(BaseBenchmarkModel):
    @staticmethod
    def add_model_options(parser_group, out_dim, modality=None):
        """
        Defining the arguments of the model. 

        Parameters
        ----------
        parser_group : argparse._ArgumentGroup
            The parser group to which model arguments must be added
        out_dim : int
            The number of classes. This is automatically set by the task choice
        modality : str, optional
            The input modality. This is automatically set by the modality choice
        """

        if modality is None:
            raise ValueError('Modality not specified')

        if modality == "eeg":
            num_chan = 16
            num_time = 4 * 128
        else:
            num_chan = 1
            if modality == "ppg":
                num_time = 6 * 128
            elif modality == "eda":
                num_time = 4 * 64
            elif modality == "resp":
                num_time = 10 * 32
            else:
                raise ValueError(f"Unknown modality: {modality}")

        # Required model arguments
        parser_group.add_argument("--num_time", default=num_time, type=int, help="Number of time steps")
        parser_group.add_argument("--num_chan", default=num_chan, type=int, help="Number of channels")
        parser_group.add_argument("--out_dim", default=out_dim, type=int,
                                  help="Size of the output.")
        

        # Custom model arguments.
        parser_group.add_argument("--predicted_class", default=0, type=int,
                                  help="The index of the class this model naively predicts.")

    def __init__(self, *, num_time, num_chan, out_dim, predicted_class):
        """
        All model arguments in the add_model_options function are 
        automatically passed as parameters to this init function. 

        Parameters
        ----------
        num_time : int
            Number of time steps for the modality of choice
        num_chan : int
            Number of channels for the modality of choice
        out_dim : int
            The number of classes. This is automatically set by the task choice 
        predicted_class : int
            An exemplary custom model argument that decides which class to predict
        """

        super().__init__()
        self.out_dim = out_dim
        self.predicted_class = predicted_class

    def forward(self, x):
        """
        Processing the input data. 

        Parameters
        ----------
        x : torch.Tensor
            A torch tensor for a single modality
        """

        batch_size = x.shape[0]
        output = torch.zeros(batch_size, self.out_dim)
        output[:, self.predicted_class] = 1
        return output


if __name__ == "__main__":
    dummy_model = UnimodalExample(
        num_time=4 * 128,
        num_chan=16,
        out_dim=2,
        predicted_class = 0
    )

    batch_size = 4
    dummy_eeg = torch.randn(batch_size, 16, 4 * 128)

    print(type(dummy_eeg))
    
    output = dummy_model(dummy_eeg)
    
    print(output)
    print(output.shape)
