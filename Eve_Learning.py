import torch
import torch.nn as nn
import torch.optim as optim
from cryptography.fernet import Fernet
from models import Eve
import random

data_set = []

def read_sentences_from_file(filename):
    """
    Reads a file and returns a list of sentences.
    Each sentence is assumed to be on a separate line.
    
    :param filename: str - path to the file
    :return: list of sentences
    """
    sentences = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            sentences = [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
    return sentences


def xor_encrypt(msg: bytes):
    key = bytes([random.randint(0, 255) for _ in msg])
    cipher = bytes([m ^ k for m, k in zip(msg, key)])
    return cipher

# Normalize a bytes object into a float tensor [0, 1]
def bytes_to_tensor(b: bytes):
    return torch.tensor(list(b), dtype=torch.float32).unsqueeze(0) / 255.0

# Denormalize output back into a bytes object
def tensor_to_bytes(t: torch.Tensor):
    t = t.squeeze(0).detach().numpy() * 255
    return bytes([int(round(x)) for x in t])

# Train Eve on multiple XOR'd examples
def train_eve_on_random_data(msg: bytes, epochs=100, batch_size=32):
    bit_length = len(msg)

    # Initialize Eve model
    eve = Eve(bit_length)
    optimizer = optim.Adam(eve.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # Train for many epochs
    for epoch in range(epochs):
        total_loss = 0.0

        for _ in range(batch_size):
            # Create a new cipher for the same message with a new random key
            cipher = xor_encrypt(msg)

            cipher_tensor = bytes_to_tensor(cipher)
            msg_tensor = bytes_to_tensor(msg)

            # Forward + Loss
            output = eve(cipher_tensor)
            loss = loss_fn(output, msg_tensor)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / batch_size
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")

    return eve

# Run and test it!
# inf = open ("android_dev.txt", "r")
# msg = bytes(inf.read(), 'utf-8')
msg = bytes(random.choice(read_sentences_from_file("android_dev.txt")), 'utf-8')
eve_model = train_eve_on_random_data(msg)

# Test Eve on a new cipher
test_cipher = xor_encrypt(msg)
test_input = bytes_to_tensor(test_cipher)
recovered = tensor_to_bytes(eve_model(test_input))

print("\nOriginal Message: ", msg)
print("Recovered by Eve: ", recovered)
print("Success: ", recovered == msg)

# Save Eve's state_dict (weights only)
torch.save(eve_model.state_dict(), "eve_model_weights.pth")
