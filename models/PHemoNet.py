# The source code was extracted from the official code base of the PHemoNet
# by Lopez et al. (https://arxiv.org/pdf/2410.00010). We made no changes to 
# the model architecture, but simply combined all components in one python 
# file and then made the moedl inherit from our base benchmark class. 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
from   torch.nn import init
import math

from models.BaseBenchmarkModel import BaseBenchmarkModel


class PHMLinear(nn.Module):
    def __init__(self, n, in_features, out_features, cuda=True):
        super(PHMLinear, self).__init__()
        self.n = n
        self.in_features = in_features
        self.out_features = out_features
        self.cuda = cuda

        self.bias = nn.Parameter(torch.Tensor(out_features))

        self.A = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, n, n))))

        self.S = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, self.out_features//n, self.in_features//n))))

        self.weight = torch.zeros((self.out_features, self.in_features))

        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)


    def kronecker_product1(self, a, b): #adapted from Bayer Research's implementation
        siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
        res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
        siz0 = res.shape[:-4]
        out = res.reshape(siz0 + siz1)
        return out

    def kronecker_product2(self):
        H = torch.zeros((self.out_features, self.in_features))
        for i in range(self.n):
            H = H + torch.kron(self.A[i], self.S[i])
        return H

    def forward(self, input):
        self.weight = torch.sum(self.kronecker_product1(self.A, self.S), dim=0)
        # self.weight = self.kronecker_product2() <- SLOWER
        input = input.type(dtype=self.weight.type())
        return F.linear(input, weight=self.weight, bias=self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
        self.in_features, self.out_features, self.bias is not None)
        
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.A, a=math.sqrt(5))
        init.kaiming_uniform_(self.S, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.placeholder)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)


class Base(nn.Module):  
    "Base for all modalities."
    def __init__(self, input_size, units=1024):
        super(Base, self).__init__() 
        self.flat = nn.Flatten()
        self.D1 = nn.Linear(input_size, units)
        self.BN1 = nn.BatchNorm1d(units)
        self.D2 = nn.Linear(units, units)
        self.BN2 = nn.BatchNorm1d(units)
        self.D3 = nn.Linear(units, units)

    def forward(self, inputs):
        x = self.flat(inputs)
        x = self.D1(x)
        x = F.relu(self.BN1(x))
        x = self.D2(x)
        x = F.relu(self.BN2(x))
        x = F.relu(self.D3(x))
        return x


class HyperFuseNet(BaseBenchmarkModel):
    @staticmethod
    def add_model_options(parser_group):
        parser_group.add_argument("--modality_units", default=[1024, 512, 512, 512], nargs="+", type=int, help="Number of units for each modality")
        parser_group.add_argument("--fusion_units", default=1024, type=int, help="Number of units in the fusion section of the model")
        parser_group.add_argument("--dim", default=4, type=int, help="Dimensionality of the number system used by the model")
        
        parser_group.add_argument("--dropout", default=0.5, type=float, help="Dropout rate")
        

    def __init__(self, *, num_time, num_chan, out_dim, modality_units, fusion_units, dim, dropout):
        super().__init__()

        fusion_in_units = sum(modality_units)

        self.eeg = Base(input_size=num_chan[0] * num_time[0], units=modality_units[0])
        self.ppg = Base(input_size=num_chan[1] * num_time[1], units=modality_units[1])
        self.eda = Base(input_size=num_chan[2] * num_time[2], units=modality_units[2])
        self.resp = Base(input_size=num_chan[3] * num_time[3], units=modality_units[3])

        self.drop = nn.Dropout(dropout)
        self.D1 = PHMLinear(dim, fusion_in_units, fusion_in_units)
        self.BN1 = nn.BatchNorm1d(fusion_in_units)
        self.D2 = PHMLinear(dim, fusion_in_units, fusion_units)
        self.BN2 = nn.BatchNorm1d(fusion_units)
        self.D3 = PHMLinear(dim, fusion_units, fusion_units//2)
        self.BN3 = nn.BatchNorm1d(fusion_units//2)
        self.D4 = PHMLinear(dim, fusion_units//2, fusion_units//4)
        self.out_3 = nn.Linear(fusion_units//4, out_dim)

    def forward(self, x):
        eeg = x[0]
        ppg = x[1]
        eda = x[2]
        resp = x[3]

        eeg_out = self.eeg(eeg)
        ppg_out = self.ppg(ppg)
        eda_out = self.eda(eda)
        resp_out = self.resp(resp)

        concat = torch.cat([eeg_out, ppg_out, eda_out, resp_out], dim=1)
        x = self.D1(concat)
        x = F.relu(self.BN1(x))
        x = self.D2(x)
        x = F.relu(self.BN2(x))
        x = self.drop(x)
        x = self.D3(x)
        x = F.relu(self.BN3(x))
        x = F.relu(self.D4(x))
        out = self.out_3(x)
        return out
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    dummy_model = HyperFuseNet(
        num_time=[4 * 128, 6 * 128, 4 * 64, 10 * 32],
        num_chan=[16, 1, 1, 1],
        out_dim=2,
        modality_units=[1024, 512, 512, 512], 
        fusion_units=1024, 
        dim=4, 
        dropout=0.5
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
    
    print(dummy_model)
    print(count_parameters(dummy_model))

    output = dummy_model(channels)
    
    print(output)
    print(output.shape)
