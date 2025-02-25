# Dummy of a multimodal model for the benchmark. 

import torch

from utils.model_util import BaseBenchmarkModel


class MultimodalDummy(BaseBenchmarkModel):
    @staticmethod
    def add_model_options(parser_group):
        """
        Defining the arguments of the model. All of these arguments are 
        automatically passed to the models init function.

        Parameters
        ----------
        parser_group : argparse._ArgumentGroup
            The parser group to which model arguments must be added
        """

        # Exemplary model arguments.
        parser_group.add_argument("--predicted_class", default=0, type=int,
                                  help="The index of the class this model naively predicts.")

    def __init__(self, *, num_time, num_chan, out_dim, predicted_class):
        """
        All model arguments in the add_model_options function are 
        automatically passed as parameters to this init function. 
        The arguments marked with (*) are predefined and automatically 
        provided by the benchmarking system.

        Parameters
        ----------
        num_time : list
            Number of time steps for each modality (*)
        num_chan : list
            Number of channels for each modality (*)
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
        x : list
            A torch tensor for each modality
        """

        batch_size = x[0].shape[0]
        output = torch.zeros(batch_size, self.out_dim)
        output[:, self.predicted_class] = 1
        return output


if __name__ == "__main__":
    """
    This function is not required by the benchmarking system. When designing a 
    custom model it merely provides a way to check if data can be passed through. 
    """
    
    dummy_model = MultimodalDummy(
        num_time=[4 * 128, 6 * 128, 4 * 64, 10 * 32],
        num_chan=[16, 1, 1, 1],
        out_dim=2,
        predicted_class = 0
    )

    batch_size = 4
    dummy_eeg = torch.randn(batch_size, 16, 4 * 128)
    dummy_ppg = torch.randn(batch_size, 1, 6 * 128)
    dummy_eda = torch.randn(batch_size, 1, 4 * 64)
    dummy_resp = torch.randn(batch_size, 1, 10 * 32)
    channels = [
        dummy_eeg,
        dummy_ppg, 
        dummy_eda,
        dummy_resp
    ]
    
    output = dummy_model(channels)
    
    print(output)
    print(output.shape)
