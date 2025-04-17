import torch
import torch.nn as nn
import torch.optim as optim
from models import Alice, Bob, Eve

# Step 1: Generate random messages and keys
def generate_batch(batch_size=128, bit_length=16):
    msg = torch.randint(0, 2, (batch_size, bit_length)).float()
    key = torch.randint(0, 2, (batch_size, bit_length)).float()
    return msg, key

# Step 2: Hyperparameters
bit_length = 16
batch_size = 128
epochs = 100

# Step 3: Load models
alice = Alice()
bob = Bob()
eve = Eve()

# Step 4: Freeze Eve
for param in eve.parameters():
    param.requires_grad = False

# Step 5: Optimizer for Alice & Bob
alice_bob_params = list(alice.parameters()) + list(bob.parameters())
optimizer = optim.Adam(alice_bob_params, lr=0.01)

# Step 6: Binary Cross Entropy Loss
loss_fn = nn.BCELoss()

# Step 7: Training loop
for epoch in range(epochs):
    # Get new data
    msg, key = generate_batch(batch_size, bit_length)

    # Alice encrypts
    alice_input = torch.cat((msg, key), dim=1)
    cipher = alice(alice_input)

    # Bob decrypts
    bob_input = torch.cat((cipher, key), dim=1)
    bob_output = bob(bob_input)

    # Eve tries to guess the message without key
    eve_output = eve(cipher)

    # Loss calculations
    bob_loss = loss_fn(bob_output, msg)
    eve_loss = loss_fn(eve_output, msg)

    # Combined loss: Reward Bob success, punish Eve success
    combined_loss = (1.5 * bob_loss) + (1.0 - eve_loss)

    # Update Alice & Bob
    optimizer.zero_grad()
    combined_loss.backward()
    optimizer.step()

    # Progress print
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1}/{epochs} | Bob loss: {bob_loss.item():.4f} | Eve loss: {eve_loss.item():.4f}")

    # Check example
    if (epoch + 1) % 50 == 0:
        print("Sample message:  ", msg[0].round())
        print("Bob's guess:     ", bob_output[0].detach().round())
        print("Eve's guess:     ", eve_output[0].detach().round())
        print("-" * 50)

with torch.no_grad():
    test_msg, test_key = generate_batch(batch_size, bit_length)
    cipher = alice(torch.cat((test_msg, test_key), dim =1))
    bob_output = bob(torch.cat((cipher, test_key), dim =1))
    eve_output = eve(cipher)
    
    bob_acc = ((bob_output.round() == test_msg).float().mean().item()) * 100
    eve_acc = ((eve_output.round() == test_msg).float().mean().item())*100
    
    print("FINAL EVALUATION:")
    print("Sample message:  ", test_msg[0].round())
    print("Bob's guess:     ", bob_output[0].detach().round())
    print("Eve's guess:     ", eve_output[0].detach().round())
    print("-" * 50)
    print(f"Final Bob accuarcy: {bob_acc:.2f} %, Final Eve accuarcy: {eve_acc:.2f}%")