from NeuronalNetwork import NN
import ActivationFunction as AF

class Test:

    __nOfExecutions = 0

    def __init__(self, nOfExecutions=7):
        self.__nOfExecutions = nOfExecutions

    def test(self):
        names = []
        directory = "FC=1000"
        names.append(directory)
        nn = NN(activationFunction=AF.rectify, nNeuronsFCLayer1=1000)
        nn.train(trainDirectory=directory, nExecutions=self.__nOfExecutions,generateStadistics=True)
        directory = "FC=650"
        names.append(directory)
        nn = NN(activationFunction=AF.rectify,nNeuronsFCLayer1=650)
        nn.train(trainDirectory=directory, nExecutions=self.__nOfExecutions,generateStadistics=True)
        directory = "FC=50"
        names.append(directory)
        nn = NN(activationFunction=AF.rectify,nNeuronsFCLayer1=50)
        nn.train(trainDirectory=directory, nExecutions=self.__nOfExecutions,generateStadistics=True)

        return self.__nOfExecutions, names
