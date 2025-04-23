import torch
import torch.nn as nn
 
# class Alice(nn.Module):
#      def __init__(self, input_size= 32, output_size=16):
#          super(Alice, self).__init__()
#          self.model = nn.Sequential(
#              nn.Linear(input_size, 32),
#              nn.ReLU(),
#              nn.Linear(32, output_size),
#              nn.Sigmoid()
#          )
#      def forward(self,x):
#          return self.model(x)
 
# class Bob(nn.Module):
#      def __init__(self, input_size=32,output_size=16):
#          super(Bob, self).__init__()
#          self.model = nn.Sequential(
#              nn.Linear(input_size, 32),
#              nn.Linear(input_size, 64),
#              nn.ReLU(),
#              nn.Linear(64,32),
#              nn.ReLU(),
#              nn.Linear(32, output_size),
#              nn.Sigmoid()
#          )
#      def forward(self,x):
#          return self.model(x)
 
# class Eve(nn.Module):
#      def __init__(self, input_size=16, output_size=16):
#          super(Eve, self).__init__()
#          self.model = nn.Sequential(
#              nn.Linear(input_size, 32),
#              nn.ReLU(),
#              nn.Linear(32, output_size),
#              nn.Sigmoid()
#          )
 
#      def forward(self, x):
#          return self.model(x)

# import torch.nn as nn

# class Alice(nn.Module):
#     def __init__(self, bit_length=16):
#         super(Alice, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(bit_length * 2, 64),
#             nn.ReLU(),
#             nn.Linear(64, bit_length),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         return self.model(x)

# class Bob(nn.Module):
#     def __init__(self, bit_length=16):
#         super(Bob, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(bit_length * 2, 64),  # 16-bit cipher + 16-bit key = 32
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, bit_length),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         return self.model(x)

class Eve(nn.Module):
    def __init__(self, bit_length):
        super(Eve, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(bit_length, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, bit_length),
            nn.Sigmoid()
        )
    def forward(self, x):
            return self.model(x)