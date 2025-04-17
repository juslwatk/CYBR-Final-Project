import torch
import torch.nn as nn
import torch.optim as optim
from models import Alice, Bob, Eve

# Generate random binary data
def generate_batch(batch_size=4, bit_length=16):
    msg = torch.randint(0, 2, (batch_size, bit_length)).float()
    key = torch.randint(0, 2, (batch_size, bit_length)).float()
    return msg, key

# Setup
bit_length = 16
batch_size = 128
epochs = 300

# build models
alice = Alice()
eve = Eve()

# optimizer for Eve
eve_optimizer = optim.Adam(eve.parameters(), lr=0.01)

# Loss funcion: how to close Eve's guess is to the actual message
loss_fn = nn.BCELoss()

# Training loop
for epoch in range(epochs):
    # Step 1: generate data
    msg, key = generate_batch(batch_size, bit_length)
    
    # Step 2: Alice encrypts
    alice_input = torch.cat((msg,key), dim = 1)
    cipher = alice(alice_input)
    
    # Step 3: Eve tries to guess the original message from the cipher
    eve_output = eve(cipher)
    
    # Step 4: Calculate the loss between Eve's guess and the original message
    loss = loss_fn(eve_output, msg)
    
    # Step 5: 
    eve_optimizer.zero_grad()
    loss.backward()
    eve_optimizer.step()
    
    # Step 6: Print progress
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1}/{epochs} | Eve's loss {loss.item():.4f}")