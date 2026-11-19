import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Residual_std_gelu(nn.Module):  #@save
    def __init__(self, input_View):
        super().__init__()
        self.Res = nn.Sequential(
            nn.Linear(input_View, input_View),
            nn.GELU(),
            nn.Linear(input_View, input_View),
            nn.GELU(),
            )

    def forward(self, X):
        return self.Res(X) + X

class Residual_1_3_gelu(nn.Module):  #@save
    def __init__(self, input_View):
        super().__init__()
        self.L1 = nn.Sequential(
            nn.Linear(input_View, input_View),
            nn.GELU(),
            )
        self.L2 = nn.Sequential(
            nn.Linear(input_View, input_View),
            nn.GELU(),
            nn.Linear(input_View, input_View),
            nn.GELU(),
            nn.Linear(input_View, input_View),
            nn.GELU(),
            )
    def forward(self, X):
        X = self.L1(X)
        Y = self.L2(X)
        return Y + X

    
class Residual_4_GELU(nn.Module):
    def __init__(self, input_View):
        super().__init__()
        self.Res = nn.Sequential(
            nn.Linear(input_View, input_View),
            nn.GELU(),
            nn.Linear(input_View, input_View),
            nn.GELU(),
            nn.Linear(input_View, input_View),
            nn.GELU(),
            nn.Linear(input_View, input_View),
            nn.GELU(),
            )

    def forward(self, X):
        return self.Res(X) + X

# Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.mole = nn.Sequential(
            nn.Linear(31, 32),
            nn.GELU(),
            Residual_std_gelu(32),
        )
        self.temp = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
        )
        self.pres = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
        )
        self.gas = nn.Sequential(
            Residual_1_3_gelu(96),
            nn.Linear(96, 1),
        )
        self.liquid = nn.Sequential(
            Residual_4_GELU(96),
            Residual_1_3_gelu(96),
            nn.Linear(96, 1),
        )
    def forward(self, x, Tr, Pr, mask):
        fea = torch.cat((self.mole(x), self.temp(Tr), self.pres(torch.log(Pr))), dim=1)
        return (1 - mask) * self.gas(fea) + mask * self.liquid(fea)
    
    @torch.jit.export
    def forward_phi(self, x, Tr, Pr):
        fea = torch.cat((self.mole(x), self.temp(Tr), self.pres(torch.log(Pr))), dim=1)
        temp_gas = self.gas(fea)
        temp_liquid = self.liquid(fea)
        mask = (Tr < 1) & (Pr < 1)
        mask_ = (Tr < 1) & (Pr >= 1)
        return torch.where(mask, torch.min(temp_gas, temp_liquid), torch.where(mask_, temp_liquid, temp_gas))

    @torch.jit.export
    def forward_background(self, x, Tr, Pr):
        return self.liquid(torch.cat((self.mole(x), self.temp(Tr), self.pres(torch.log(Pr))), dim=1))
    
    @torch.jit.export
    def forward_phi_only(self, x, Tr, Pr):
        return self.gas(torch.cat((self.mole(x), self.temp(Tr), self.pres(torch.log(Pr))), dim=1))