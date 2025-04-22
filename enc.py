from cryptography.fernet import Fernet

#basic example of reading a msg from a file, encrypting it, and decrypting it
#generates key
key = Fernet.generate_key()
cipher_suite = Fernet(key)

#reads line from file and formats it for encryption
inf = open ("msg.txt", "r")
msg = bytes(inf.read(), 'utf-8')

#encrypts message
cipher = cipher_suite.encrypt(msg)

#decrypts message
orig = cipher_suite.decrypt(cipher)
orig = orig.decode()
inf.close()

#Key generation
class GenKey:
    def __init__():
        key = Fernet.generate_key()
    def foward():
        return key

#Encryption class
class Enc():
    def __init__(self, key):
        cipher_suite = Fernet(key)

        inf = open ("msg.txt", "r")
        msg = bytes(inf.readline(), 'utf-8')
        cipher = cipher_suite.encrypt(msg)
    
    def forward():
        return cipher

class Dec():
    def __init__(self, orig):
        cipher_suite = Fernet(key)
        orig = cipher_suite.decrypt(cipher)
        orig = orig.decode()
    def forward(self, orig):
        return self.orig
