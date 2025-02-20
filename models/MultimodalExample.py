# Example of a multimodal model for the benchmark. 

import torch

from utils.model_util import BaseBenchmarkModel


class MultimodalExample(BaseBenchmarkModel):
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
            The input modality. In multimodal settings this should not be used and is always None.
        """

        # Required model arguments
        parser_group.add_argument("--num_time", default=[4 * 128, 6 * 128, 4 * 64, 10 * 32],
                          type=int, nargs="+", help="Number of time steps for each modality")
        parser_group.add_argument("--num_chan", default=[16, 1, 1, 1], type=int, nargs="+",
                          help="Number of channels for each modality")
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
        num_time : list
            Number of time steps for each modality
        num_chan : list
            Number of channels for each modality
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
        x : list
            A torch tensor for each modality
        """

        batch_size = x[0].shape[0]
        output = torch.zeros(batch_size, self.out_dim)
        output[:, self.predicted_class] = 1
        return output


if __name__ == "__main__":
    dummy_model = MultimodalExample(
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
