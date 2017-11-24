from Test import Test
import ActivationFunction as AF

import NeuronalNetwork as NN
nn = NN.NN(nNeuronsConvLayer1=8,nNeuronsConvLayer2=8,nNeuronsConvLayer3=8,nNeuronsFCLayer1=50,nNeuronsFCLayer2=2,activationFunction=AF.rectify)
nn.configureConvNetParameters(kernelSize=6,sizeImage=256,maxPoolSize=2)
nn.train(trainDirectory="test",batch=10,epochs=20)