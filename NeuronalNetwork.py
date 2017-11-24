import numpy as np
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
import ActivationFunction as AF
import GradientDescent as GD
from Directory import createDirectory
import ModelPersistence
import Graphics
from theano.tensor.nnet import conv
import pylab
from PIL import Image
import os
import shutil
import math

obj= ["0","1","2","3","4","5","6","7","8","9"]
trainingPath = "Training/"

class NN:

    w = None
    w2 = None
    w3 = None
    w4 = None
    w_o = None
    activationFunction = AF.rectify
    __LearningCurve=True
    __Histograms=False
    __Features=False
    __LayerStadistics=False
    __nw=0
    __nw2=0
    __nw3=0
    __nw4=0
    __nw_o=0
    __sizeImage = 256
    __kernelSize = 3
    __maxPoolSize = 2
    __ajustKernel2FullConnected = 31
    __channels = 1

    # initialize weights and size of layers
    def __init__(self,sizeImage=256, nNeuronsConvLayer1=32,nNeuronsConvLayer2=64,nNeuronsConvLayer3=128,nNeuronsFCLayer1=625,nNeuronsFCLayer2=2,
                 activationFunction=AF.rectify):
        self.activationFunction = activationFunction
        self.__nw = nNeuronsConvLayer1
        self.__nw2 = nNeuronsConvLayer2
        self.__nw3 = nNeuronsConvLayer3
        self.__nw4 = nNeuronsFCLayer1
        self.__nw_o = nNeuronsFCLayer2
        self.__sizeImage = sizeImage
        self.__initializeWeights()

    def configureConvNetParameters(self, sizeImage, kernelSize, maxPoolSize,channels=1):
        self.__sizeImage=sizeImage
        self.__kernelSize=kernelSize
        self.__maxPoolSize=maxPoolSize
        self.__channels = channels
        n1 = math.ceil((sizeImage-kernelSize+1)/maxPoolSize)
        n2 = math.ceil((n1-kernelSize+1)/maxPoolSize)
        n3 = math.ceil((n2-kernelSize+1)/maxPoolSize)
        self.__ajustKernel2FullConnected=n3

    # Forward propagation
    def __model(self,X, w, w2, w3, w4, w_o, p_drop_conv, p_drop_hidden):

        # Dropout is a way to reduce overfitting
        def dropout(X, p=0.):
            if p > 0:
                srng = RandomStreams()
                retain_prob = 1 - p
                X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
                X /= retain_prob
            return X

        l1a = self.activationFunction(conv2d(X, w, border_mode='full'))
        l1 = max_pool_2d(l1a, (2, 2),ignore_border=False)
        l1 = dropout(l1, p_drop_conv)

        l2a = self.activationFunction(conv2d(l1, w2))
        l2 = max_pool_2d(l2a, (2, 2),ignore_border=False)
        l2 = dropout(l2, p_drop_conv)

        l3a = self.activationFunction(conv2d(l2, w3))
        l3b = max_pool_2d(l3a, (2, 2),ignore_border=False)
        l3 = T.flatten(l3b, outdim=2)
        l3 = dropout(l3, p_drop_conv)

        
        l4 = self.activationFunction(T.dot(l3, w4))
        l4 = dropout(l4, p_drop_hidden)

        pyx = AF.softmax(T.dot(l4, w_o))
        return l1, l2, l3b, l4, pyx

    # this method load trainset and testset
    def __loadData(self):
        import Input_data
        trX, trY, teX, teY = Input_data.get_data()
        self.__sizeImage = trX.shape[3]
        self.__nw_o = trY.shape[1]
        return trX, trY, teX, teY

    def __initializeWeights(self):
        # Translate a list into a Numpy array
        def floatX(X):
            return np.asarray(X, dtype=theano.config.floatX)

        # Initialize the weights of every layer. The "*" means converting a
        # list in a sequence of parameters
        def init_weights(shape):
            return theano.shared(floatX(np.random.randn(*shape) * 0.01))

        self.w = init_weights((self.__nw, self.__channels, self.__kernelSize, self.__kernelSize))
        self.w2 = init_weights((self.__nw2, self.__nw, self.__kernelSize, self.__kernelSize))
        self.w3 = init_weights((self.__nw3, self.__nw2, self.__kernelSize, self.__kernelSize))
        self.w4 = init_weights((self.__nw3 * self.__ajustKernel2FullConnected**2, self.__nw4))
        self.w_o = init_weights((self.__nw4, self.__nw_o))

    def __generateStadistics(self,path, teX, teY):
        print "     Generating stadistics..."
        if (self.__LearningCurve):
            print "         Cost and hits mean..."
            cost = path+"Cost.txt"
            hit = path+"HitPercentage.txt"
            Graphics.plotLearningCurve([cost],"Cost", path,True)
            Graphics.plotLearningCurve([hit],"Hit Percentage", path,True)
        if (self.__Histograms):
            print "         Histograms..."
            Graphics.plotLayerActivationHistogram([self.w.eval()], "Conv1", path, True)
            Graphics.plotLayerActivationHistogram([self.w2.eval()], "Conv2", path, True)
            Graphics.plotLayerActivationHistogram([self.w3.eval()], "Conv3", path, True)
            Graphics.plotLayerActivationHistogram([self.w4.eval()], "FC1", path, True)
            Graphics.plotLayerActivationHistogram([self.w_o.eval()], "Output", path, True)
        if (self.__Features):
            print "         Features..."
            Graphics.plotConvolutionLayerKernels(self.w.eval(), "Conv1", path)
            Graphics.plotConvolutionLayerKernels(self.w2.eval(), "Conv2",path)
            Graphics.plotConvolutionLayerKernels(self.w3.eval(), "Conv3",path)
        if (self.__LayerStadistics):
            print "         Calculating layer stadistics..."
            X = T.ftensor4()
            l1, l2, l3, l4, py_x = self.__model(X, self.w, self.w2, self.w3, self.w4, self.w_o, 0., 0.)
            activations = theano.function(inputs=[X], outputs=[l1, l2, l3, l4, py_x], allow_input_downcast=True)
            Graphics.plotLayerStadistics(teX, activations=activations , exportPath=path)
        print

    def configureStadistics(self, learningCurve=True, histograms=False, features=False, layerStadistics=False):
        self.__Histograms=histograms
        self.__LearningCurve=learningCurve
        self.__Features=features
        self.__LayerStadistics=layerStadistics

    def calculePercentageOfHits(self):
        trX, trY ,teX, teY = self.__loadData()
        print "Data Load"
        X = T.ftensor4()
        Y = T.fmatrix()
        py_x = self.__model(X, self.w, self.w2, self.w3, self.w4, self.w_o, 0., 0.)[4]
        y_x = T.argmax(py_x, axis=1)
        predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=False)
        print "Percentage of hits: "+str(np.mean(np.argmax(teY, axis=1) == predict(teX)))

    def load_model_weights(self,filePath='best_model.np'):
        print "Loading model..."
        params = ModelPersistence.loadModel(filePath)
        self.w = theano.shared(params[0])
        self.w2 = theano.shared(params[1])
        self.w3 = theano.shared(params[2])
        self.w4 = theano.shared(params[3])
        self.w_o = theano.shared(params[4])
        print "Model load\n"

    def prediction(self, imagePath):
        X = T.ftensor4()
        l1, l2, l3, l4, py_x = self.__model(X, self.w, self.w2, self.w3, self.w4, self.w_o, 0., 0.)
        y_x = T.argmax(py_x, axis=1)
        predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)
        img = Image.open(open(imagePath))
        img = np.asarray(img, dtype='float64') / 255.
        img2 = np.array([img])
        img_ = img2.reshape(1, 1, 28, 28)
        out = predict(img_)
        print "Prediction = ",out

    def visualizeConvLayer(self, imagePath, exportPath="convFilters", gray=True):
        # create theano function to compute filtered images
        input = T.tensor4(name='input')
        w1 = theano.shared( np.asarray(self.w.eval(),dtype=input.dtype), name ='W')
        w2 = theano.shared( np.asarray(self.w2.eval(),dtype=input.dtype), name ='W2')
        w3 = theano.shared( np.asarray(self.w3.eval(),dtype=input.dtype), name ='W3')
        outputw1 = self.activationFunction(conv.conv2d(input, w1))
        outputw2 = self.activationFunction(conv.conv2d(outputw1, w2))
        outputw3 = self.activationFunction(conv.conv2d(outputw2, w3))
        features1 = w1.eval().shape[0]; features2 = w2.eval().shape[0]; features3 = w3.eval().shape[0]
        f = theano.function([input], [outputw1,outputw2,outputw3])

        #create the directories
        if os.path.exists(exportPath)==0:
            os.mkdir(exportPath)
        exportPath1 = exportPath+"/"+"Filter Conv1"+"/";
        if os.path.exists(exportPath1)==0:
            os.mkdir(exportPath1)
        exportPath2 = exportPath+"/"+"Filter Conv2"+"/";
        if os.path.exists(exportPath2)==0:
            os.mkdir(exportPath2)
        exportPath3 = exportPath+"/"+"Filter Conv3"+"/";
        if os.path.exists(exportPath3)==0:
            os.mkdir(exportPath3)

        # open image
        img = Image.open(open(imagePath))

        # dimensions are (height, width, channel)
        img = np.asarray(img, dtype='float64') / 255.

        # put image in 4D tensor of shape (1, channels, height, width)
        img2 = np.array([img])
        img_ = img2.reshape(1, 1, 28, 28)
        filtered_img1,filtered_img2,filtered_img3 = f(img_)

        pylab.figure(figsize=(1,1))
        pylab.axis('off')
        # plot original image and features
        shutil.copy2(imagePath, exportPath+'/Real Image.png')
        if (gray): pylab.gray();
        for i in range(0, features1):
            pylab.imshow(filtered_img1[0, i, :, :],aspect='auto');
            pylab.savefig(exportPath1+'Conv1 Filter '+str(i)+'.png')
        for i in range(0, features2):
            pylab.imshow(filtered_img2[0, i, :, :],aspect='auto');
            pylab.savefig(exportPath2+'Conv2 Filter '+str(i)+'.png')
        for i in range(0, features3):
            pylab.imshow(filtered_img3[0, i, :, :],aspect='auto');
            pylab.savefig(exportPath3+'Conv3 Filter '+str(i)+'.png')

        print "VisualizeConvLayer Complete!"

    def __sensitivity(self,teX,teY,predict):
        teeY0 = []
        teeY1 = []
        teeX0 = []
        teeX1 = []
        for i in range(teY.shape[0]):
            if (np.argmax(teY[i,:]) == 0): teeY0.append(teY[i,:]); teeX0.append(teX[i,:]);
            if (np.argmax(teY[i,:]) == 1): teeY1.append(teY[i,:]); teeX1.append(teX[i,:]);
        teeY0 = np.array(teeY0); teeX0 = np.array(teeX0);
        teeY1 = np.array(teeY1); teeX1 = np.array(teeX1);
        print "         Sensitivity: " + str(np.mean(np.argmax(teeY1, axis=1) == predict(teeX1)))
        print "         Especificity: " + str(np.mean(np.argmax(teeY0, axis=1) == predict(teeX0)))

    def train(self, epochs = 10, batch=128, lr=0.001 ,p_drop_conv=0.2, p_drop_hidden=0.5 , saveModel=False,
              trainDirectory='Default', nExecutions=1, generateStadistics=False):
        print "\nLoading Data..."
        trX, trY ,teX, teY = self.__loadData()

        for x in range(0,nExecutions):
            X = T.ftensor4()
            Y = T.fmatrix()
            n1, n2, n3, n4, noise_py_x = self.__model(X, self.w, self.w2, self.w3, self.w4, self.w_o, p_drop_conv, p_drop_hidden)
            cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
            params = [self.w, self.w2, self.w3, self.w4, self.w_o]
            updates = GD.RMSprop(cost, params, lr)
            executeTrain = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
            l1, l2, l3, l4, py_x = self.__model(X, self.w, self.w2, self.w3, self.w4, self.w_o, 0., 0.)
            y_x = T.argmax(py_x, axis=1)
            predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

            self.__initializeWeights()
            createDirectory(trainingPath)
            path = trainingPath+str(trainDirectory)+"/"
            createDirectory(path)
            path = path+"Execution"+str(x)+"/"
            createDirectory(path)

            # Start train
            f = open(str(path)+'Cost.txt', 'w')
            f.write('Cost (Execution'+str(x)+')\n')
            f2 = open(str(path)+'HitPercentage.txt', 'w')
            f2.write('Hit Percentage (Execution'+str(x)+')\n')
            save = False
            m = 0
            maxMean = 0
            print "\nExecution "+str(x)
            print "---------------------------------------"
            for i in range(epochs):
                print " Iteration "+str(i)+":"
                error = 0
                for start, end in zip(range(0, len(trX), batch), range(batch, len(trX), batch)):
                    error = executeTrain(trX[start:end], trY[start:end])
                mean = np.mean(np.argmax(teY, axis=1) == predict(teX))
                print "     Mean: "+str(mean)
                self.__sensitivity(teX,teY,predict)
                print "     Cost: "+str(error)
                if (mean<m): save=True; maxMean=mean
                if saveModel and save and mean>maxMean: ModelPersistence.saveModel(params,filePath=path+"best_model.np"); maxMean=mean
                m = mean
                f.write(str(error)+"\n")
                f.flush()
                f2.write(str(m)+"\n")
                f2.flush()
            f.close()
            f2.close()
            print

            if generateStadistics: self.__generateStadistics(path, teX, teY)

        # Plot all the executions together
        if (nExecutions>1):
            costs = [(trainingPath+str(trainDirectory)+"/Execution"+str(x)+"/Cost.txt" ) for x in range(nExecutions)]
            means = [(trainingPath+str(trainDirectory)+"/Execution"+str(x)+"/HitPercentage.txt" ) for x in range(nExecutions)]
            Graphics.plotLearningCurve(costs,"Costs",trainingPath+str(trainDirectory)+"/",True)
            Graphics.plotLearningCurve(means,"Hit Percentages",trainingPath+str(trainDirectory)+"/",True)