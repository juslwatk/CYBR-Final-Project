import torch
import numpy as np
from models import Alice, Bob, Eve

def generate_batch(batch_size=4, bit_length=16):
    msg = torch.randint(0, 2, (batch_size, bit_length)).float()
    key = torch.randint(0, 2, (batch_size, bit_length)).float()
    return msg, key

alice = Alice()
bob = Bob()
eve = Eve()

msg, key = generate_batch()

print(msg)
print(key)