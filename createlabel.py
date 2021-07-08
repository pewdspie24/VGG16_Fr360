import os

with open('.labels.txt', "a") as f:
    for i in os.listdir("./fruits-360/Training"):
        f.write(i+'\n')