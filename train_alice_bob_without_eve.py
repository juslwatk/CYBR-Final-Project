import torch
import torch.nn as nn
import torch.optim as optim
from models import Alice, Bob

# Step 1: Generate random binary messages and keys
def generate_batch(batch_size=128, bit_length=16):
    msg = torch.randint(0, 2, (batch_size, bit_length)).float()
    key = torch.randint(0, 2, (batch_size, bit_length)).float()
    return msg, key

# Step 2: Hyperparameters
bit_length = 16
batch_size = 128
epochs = 300

# Step 3: Create Alice and Bob
alice = Alice()
bob = Bob()

# Step 4: Create a single optimizer for both Alice and Bob
alice_bob_params = list(alice.parameters()) + list(bob.parameters())
optimizer = optim.Adam(alice_bob_params, lr=0.01)

# Step 5: Binary Cross Entropy Loss
loss_fn = nn.BCELoss()

# Step 6: Training loop
for epoch in range(epochs):
    # Generate new messages and keys
    msg, key = generate_batch(batch_size, bit_length)

    # Alice encrypts the message using the key
    alice_input = torch.cat((msg, key), dim=1)
    cipher = alice(alice_input)

    # Bob decrypts the cipher using the key
    bob_input = torch.cat((cipher, key), dim=1)
    bob_output = bob(bob_input)

    # Compare Bob's output to the original message
    bob_loss = loss_fn(bob_output, msg)

    # Update both models
    optimizer.zero_grad()
    bob_loss.backward()
    optimizer.step()

    # Print every 10 epochs + sample every 50
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1}/{epochs} | Bob loss: {bob_loss.item():.4f}")
    
    if (epoch + 1) % 50 == 0:
        print("Sample message:  ", msg[0].round())
        print("Bob's guess:     ", bob_output[0].detach().round())
        print("-" * 50)
