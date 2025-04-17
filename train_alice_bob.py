import torch
import torch.nn as nn
import torch.optim as optim
from models import Alice, Bob, Eve

# Generate random binary data for key and message
def generate_batch(batch_size=4, bit_length=16):
    msg = torch.randint(0, 2, (batch_size, bit_length)).float()
    key = torch.randint(0, 2, (batch_size, bit_length)).float()
    return msg, key

# Hyperparameters
bit_length = 16
batch_size = 128
epochs = 300

# Create Models
alice = Alice()
bob = Bob()
eve = Eve()

# Freeze Eve
for param in eve.parameters():
    param.requires_grad = False
    
# Optimizer for Alice and Bob together
alice_bob_params = list(alice.parameters()) + list(bob.parameters())
optimizer = optim.Adam(alice_bob_params, lr=.001)

# Loss function
loss_fn = nn.BCELoss()

# Training loop
for epoch in range(epochs):
    # Step 1: Generate message and key
    msg, key = generate_batch(batch_size, bit_length)
    
    # Step 2: Alice encrypts
    alice_input = torch.cat((msg,key), dim=1)
    cipher = alice(alice_input)
    
    # Step 3: Bob encrypts
    bob_input = torch.cat((cipher, key), dim=1)
    bob_output = bob(bob_input)
    
    # Step 4: Eve tries to guess the message
    eve_output = eve(cipher)
    
    # Step 5: Calculate the losses
    bob_loss = loss_fn(bob_output, msg)
    eve_loss = loss_fn(eve_output,msg)
    
    # Combined loss: Help Bob, Confuse Eve
    combined_loss = (1.5 * bob_loss) + (1.0 - eve_loss)
    
    # Step 6: Update Alice and Bob
    optimizer.zero_grad()
    combined_loss.backward()
    optimizer.step
    
    # Step 7: Print results
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1}/{epochs} | Bob loss: {bob_loss.item():.4f} | Eve loss: {eve_loss.item():.4f}")
    
    if (epoch +1) % 50 == 0 or epoch == 0:
        print(f"Sample message:     ", msg[0].round())
        print(f"Bob's guess:     ", bob_output[0].detach().round())
        print(f"Eve's guess:     ", eve_output[0].detach().round())
        print("-" * 50)

    