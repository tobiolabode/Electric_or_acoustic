import os

i = 0
for filename in os.listdir("Data/electric guitar/"):
    dst = "electric-guitar" + str(i) + ".jpg"
    src = "Data/electric guitar/" + filename
    dst = "Data/electric guitar/" + dst
    os.rename(src, dst)
    i += 1
