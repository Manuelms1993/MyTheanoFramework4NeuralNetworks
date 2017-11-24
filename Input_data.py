import numpy as np
import cPickle
import gzip
import random

def shuffleData(trX, trY, teX, teY):

    teN = range(teX.shape[0])
    random.shuffle(teN)
    TEX=[]; TEY=[]
    for i in range(teX.shape[0]):
        TEX.append(teX[teN[i]])
        TEY.append(teY[teN[i]])
    teX = np.array(TEX)
    teY = np.array(TEY)

    trN = range(trX.shape[0])
    random.shuffle(trN)
    TRX=[]; TRY=[]
    for i in range(trX.shape[0]):
        TRX.append(trX[trN[i]])
        TRY.append(trY[trN[i]])
    trX = np.array(TRX)
    trY = np.array(TRY)

    for i in range(teX.shape[0]-1):
        if random.random()<0.9:
            index = random.randint(0,trX.shape[0]-1)
            if teY[i][1]==trY[index][1]:
                teX[i],trX[index] = trX[index],teX[i]
                teY[i],trY[index] = trY[index],teY[i]

    return trX, trY, teX, teY


def get_data():

    def unpickle(file):
        fo = open(file, 'rb')
        dictionary = cPickle.load(fo)
        fo.close()
        return dictionary

    def one_hot(x, n):
        if type(x) == list:
            x = np.array(x)
        x = x.flatten()
        o_h = np.zeros((len(x), n))
        o_h[np.arange(len(x)), x-1] = 1
        return o_h

    faces_train = unpickle("../../Datasets/short/data_objects")
    faces_test = unpickle("../../Datasets/short/test_objects")

    trX = faces_train["data"]
    trY = faces_train["labels"]

    teX = faces_test["data"]
    teY = faces_test["labels"]

    trY = one_hot(trY, 2)
    teY = one_hot(teY, 2)

    trX = trX.reshape(-1, 1, 256, 256)
    teX = teX.reshape(-1, 1, 256, 256)

    trX, trY, teX, teY = shuffleData(trX, trY, teX, teY)

    print "-------"
    print trX.shape
    print trY.shape
    print teX.shape
    print teY.shape
    print "-------"

    return trX, trY, teX, teY


def get_data_mnist():

    def one_hot(x, n):
        if type(x) == list:
            x = np.array(x)
        x = x.flatten()
        o_h = np.zeros((len(x), n))
        o_h[np.arange(len(x)), x] = 1
        return o_h

    f = gzip.open('../../Datasets/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    trX, trY = train_set
    teX, teY = test_set
    f.close()
    trY = one_hot(trY, 10)
    teY = one_hot(teY, 10)
    trX = trX.reshape(-1, 1, 28, 28)
    teX = teX.reshape(-1, 1, 28, 28)
    return trX, trY, teX, teY
