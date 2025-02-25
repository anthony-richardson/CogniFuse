# Dummy of a unimodal model for the benchmark. 

import torch
from torch import nn 

from utils.model_util import BaseBenchmarkModel


class UnimodalDummy(BaseBenchmarkModel):
    @staticmethod
    def add_model_options(parser_group, modality):
        """
        Defining custom arguments for the model. All of these arguments are 
        automatically passed to the models init function. 

        Parameters
        ----------
        parser_group : argparse._ArgumentGroup
            The parser group to which model arguments must be added
        modality : str
            The input modality. This is automatically set by the modality choice. 
            Can be used to define different behaviour depending on the modality. 
        """

        # Exemplary model argument.
        parser_group.add_argument("--predicted_class", default=0, type=int,
                                  help="The index of the class this model naively predicts.")

    def __init__(self, num_time, num_chan, out_dim, predicted_class):
        """
        All model arguments in the add_model_options function are 
        automatically passed as parameters to this init function. 
        The arguments marked with (*) are predefined and automatically 
        provided by the benchmarking system.

        Parameters
        ----------
        num_time : int
            Number of time steps for the modality of choice (*)
        num_chan : int
            Number of channels for the modality of choice (*)
        out_dim : int
            The number of classes. This is automatically set by the task choice (*)
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
    """
    This function is not required by the benchmarking system. When designing a 
    custom model it merely provides a way to check if data can be passed through. 
    """

    dummy_model = UnimodalDummy(
        num_time=4 * 128,
        num_chan=16,
        out_dim=2,
        predicted_class=0
    )

    batch_size = 4
    dummy_eeg = torch.randn(batch_size, 16, 4 * 128)

    print(type(dummy_eeg))
    
    output = dummy_model(dummy_eeg)
    
    print(output)
    print(output.shape)
