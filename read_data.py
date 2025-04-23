import onetimepad
with open("android_dev.txt","r") as input:
    with open("data.txt", "w") as output:
        for line in input:
            cipher = onetimepad.encrypt(line,'1234567')
            output.write(cipher)
            output.write("/n")
