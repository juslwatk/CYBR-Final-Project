import torch
import torch.nn as nn

class Alice(nn.Module):
    def __init__(self, input_size= 32, output_size=16):
        super(Alice, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.model(x)
    
class Bob(nn.Module):
    def __init__(self, input_size=32,output_size=16):
        super(Bob, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.model(x)
    
class Eve(nn.Module):
    def __init__(self, input_size=16, output_size=16):
        super(Eve, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)
    

