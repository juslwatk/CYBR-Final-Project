import torch
import numpy as np
from models import Alice, Bob, Eve

# Function to generate random message and key
def generate_batch(batch_size=4, bit_length=16):
    msg = torch.randint(0, 2, (batch_size, bit_length)).float()
    key = torch.randint(0, 2, (batch_size, bit_length)).float()
    return msg, key

# Contructors of models
alice = Alice()
bob = Bob()
eve = Eve()

# Constructors of Key and Message
msg, key = generate_batch()

# Alice encrypts msg using key
alice_input = torch.cat((msg, key), dim=1)
cipher = alice(alice_input)

# Bob decrypts
bob_input = torch.cat((msg,key), dim=1)
bob_output_ = bob(bob_input)

# Eve then guesses
eve_output = eve(cipher)

# print everything owt
print("="*80)
print("ORIGINAL MESSAGE:\n", msg)
print("="*80)
print("KEY:\n", key)
print("="*80)
print("CIPHERTEXT FROM ALICE:\n", cipher)
print("="*80)
print("DECRYPTED TEXT FROM BOB:\n", bob_output_)
print("="*80)
print("EVE'S GUESSED OUTPUT\n", eve_output)
print("="*80)