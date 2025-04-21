#basic example of reading a msg from a file, encrypting it, and decrypting it
from cryptography.fernet import Fernet

key = Fernet.generate_key()
cipher_suite = Fernet(key)

inf = open ("msg.txt", "r")
msg = bytes(inf.read(), 'utf-8')
print(msg)
cipher = cipher_suite.encrypt(msg)
print(cipher)
orig = cipher_suite.decrypt(cipher)
orig = orig.decode()
print(orig)
inf.close()

#Key generation
class GenKey():
    key = Fernet.generate_key()

#Encryption class
class Enc(key):
    cipher_suite = Fernet(key)

    inf = open ("msg.txt", "r")
    msg = bytes(inf.readline(), 'utf-8')
    cipher = cipher_suite.encrypt(msg)
    return cipher

class Dec(key, cipher):
    cipher_suite = Fernet(key)
    orig = cipher_suite.decrypt(cipher)
    orig = orig.decode()
    return orig
