from abc import ABC, abstractmethod
from torch import nn

# This is the base for all models in the benchmark.
# Since the benchmark only supports the use of pytorch models,
# this base class also restrict the user to such models.
class BaseBenchmarkModel(nn.Module, ABC):
    @staticmethod
    @abstractmethod
    def add_model_options(parser_group): 
        pass
    