import torch
import torch.nn as nn
import torch.optim as optim
from models import Alice, Bob, Eve

def generate_batch(batch_size=128, bit_length=16):
    msg = torch.randint(0, 2, (batch_size, bit_length)).float()
    key = torch.randint(0, 2, (batch_size, bit_length)).float()
    return msg, key


# Model initialization
alice = Alice()
bob = Bob()
eve = Eve()

# Separate optimizers
alice_bob_params = list(alice.parameters()) + list(bob.parameters())
optim_ab = optim.Adam(alice_bob_params, lr=0.001)
optim_eve = optim.Adam(eve.parameters(), lr=0.001)

# Loss function
loss_fn = nn.BCELoss()

# Training settings
epochs = 300
bit_length = 16
batch_size = 128
k = 1  # number of Eve steps per epoch

pretrain_epochs = 100
for epoch in range(pretrain_epochs):
    msg, key = generate_batch(batch_size, bit_length)
    
    # Forward pass through Alice
    with torch.no_grad():
        cipher = alice(torch.cat((msg, key), dim=1)) # Freeze Alice during pretraining
    
    # Eve tries to decode without the key    
    eve_output = eve(cipher)
    eve_loss = loss_fn(eve_output, msg)
    
    # Update Eve
    optim_eve.zero_grad()
    eve_loss.backward()
    optim_eve.step()
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
         print(f"[Eve Pretraining] Epoch {epoch+1}/{pretrain_epochs} - Loss: {eve_loss.item():.4f}")
    
        
adv_epochs = 300
for epoch in range(adv_epochs):
    
    # Alice encrypts
    msg, key = generate_batch(batch_size, bit_length)
    cipher = alice(torch.cat((msg, key), dim=1))
        
    # Bob decrypts
    bob_input = torch.cat((cipher, key), dim=1)
    bob_output = bob(bob_input)
    
    # Eve tries to decode cipher
    eve_output = eve(cipher)
    
    # Losses
    bob_loss = loss_fn(bob_output,msg)
    eve_loss = loss_fn(eve_output, msg)
    
    # Combined Loss
    combined_loss = bob_loss + (1.0 - eve_loss)
    
    # Update Alice & Bob only
    ab_optimizer = optim.Adam(list(alice.parameters()) + list(bob.parameters()), lr=0.001)
    ab_optimizer.zero_grad()
    combined_loss.backward()
    ab_optimizer.step()
    
    # Also train Eve during adversarail phase
    eve_output = eve(cipher.detach()) # Detached to prevent gradients
    eve_loss = loss_fn(eve_output, msg)
    optim_eve.zero_grad()
    eve_loss.backward()
    optim_eve.step()
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
         print(f"[Adversarial] Epoch {epoch + 1}/{adv_epochs} | Bob Loss: {bob_loss.item():.4f} | Eve Loss: {eve_loss.item():.4f}")
        
    with torch.no_grad():
        test_msg, test_key = generate_batch(batch_size, bit_length)
        
        # Alice encrypts the test messages
        cipher = alice(torch.cat((test_msg, test_key), dim=1))
        
        # Bob and Eve try to decode
        bob_output = bob(torch.cat((cipher, test_key), dim=1))
        eve_output = eve(cipher)

        # Calculate accuracy
        bob_acc = ((bob_output.round() == test_msg).float().mean().item()) * 100
        eve_acc = ((eve_output.round() == test_msg).float().mean().item()) * 100

        # Print results
        print("\n🧪 FINAL EVALUATION")
        print("-" * 50)
        print("Original Message: ", test_msg[0].round())
        print("Bob's Output:     ", bob_output[0].round())
        print("Eve's Output:     ", eve_output[0].round())
        print("-" * 50)
        print(f"Bob Accuracy: {bob_acc:.2f}%")
        print(f"Eve Accuracy: {eve_acc:.2f}%")
