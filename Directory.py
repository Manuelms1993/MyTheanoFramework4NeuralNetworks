import os

def createDirectory(path):
    if os.path.exists(path)==0:
        os.mkdir(path)

def removeFile(path):
    if not os.path.exists(path)==0:
        os.remove(path)
