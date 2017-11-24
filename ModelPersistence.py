import numpy

def saveModel(params,filePath='best_model.np'):
    f = open(filePath,'wb')
    a = []
    for i in range(len(params)):
        a.append(params[i].get_value())
    numpy.save(f, a)
    f.close()

def loadModel(filePath='best_model.np'):
    f = open(filePath,'rb')
    params = numpy.load(f)
    f.close()
    return params