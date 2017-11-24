import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
import os

color = ["b", "g", "r", "c", "m", "y", "k"]
plt.ioff()

# @layer is a first convolution layer weights in black and white
def plotConvolutionLayerKernels(w, layerName, exportPath):
    plt.figure()
    exportPath = exportPath+"Kernel "+layerName+"/"
    if os.path.exists(exportPath)==0: os.mkdir(exportPath)
    for i in range((w.shape[0])):
        img = np.zeros((3,3))
        for j in range((w.shape[2])):
            for k in range((w.shape[3])):
                img[j][k] = w[i][0][j][k]
        plt.imshow(img)
        plt.savefig(exportPath+'feature'+str(i)+'.png')
        image = Image.open(exportPath+'feature'+str(i)+'.png')
        imagenbw = image.convert("L")
        imagenbw.thumbnail((800, 799), Image.ANTIALIAS)
        imagenbw.save(exportPath+'feature'+str(i)+'.png')

# @layers is a vector of activation neurons values in every layer
# If @exportPlotInPNG is True, function save the plot and doesn't show the graphic
def plotLayerActivationHistogram(layers,exportName="NeuronLayerActivation", exportPath="" ,exportPlotInPNG=False,):
    f, p = plt.subplots(len(layers),1,sharex=True,figsize=(20,10),squeeze=False)
    bins = 20
    y = [x for x in range(20)]
    for i in range(len(layers)):
        x = np.reshape(layers[i], (1,np.product(layers[i].shape)))[0]
        hist, b = np.histogram(x, bins=bins)
        p[i,0].bar(y,hist, width=1, color=color[i if i<7 else 1])
        p[i,0].axis([0,bins,0,hist[1]])
    if (exportPlotInPNG):
        exportPath = exportPath +"Histograms/"
        if os.path.exists(exportPath)==0: os.mkdir(exportPath)
        plt.savefig(exportPath+exportName+".png")
    else:
        plt.show()

# @listFile is a vector of files names ['file1','file2',...]
# If @exportPlotInPNG is True, function save the plot and doesn't show the graphic
# The first line in the file should be the legend that you want see at plot
def plotLearningCurve(listFile, exportName="Graphic", exportPath="", exportPlotInPNG=False):

    def readFile(filePath):
        costs = []
        f = open(filePath, 'r')
        legend = f.readline().split('\n')[0]
        for line in f:
            costs.append(float(line.split('\n')[0]))
        f.close()
        return costs,legend

    if (len(listFile)>7):
        print "Too many files."
        return

    if (len(listFile)==0):
        print "EMPTY VECTOR!, I need at least one file."
        return

    plt.figure(num=None, figsize=(20, 15), dpi=60, facecolor='w', edgecolor='k')
    plt.xlabel("Iterations")
    plt.ylabel(exportName)
    plt.title("Learning")
    for i in range(len(listFile)):
        costs,legend = readFile(listFile[i])
        plt.plot(costs,color[i]+'-',linewidth=2.0)
        # plt.text(np.argmax(costs), np.max(costs) + 0.02, 'Max = '+str(np.max(costs)), fontsize = 10, color=color[i],
        #          horizontalalignment='center', verticalalignment='center')
        # plt.text(np.argmin(costs), np.min(costs) - 0.04, 'Min = '+str(np.min(costs)), fontsize = 10, color=color[i],
        #          horizontalalignment='center', verticalalignment='center')
    # plt.legend(loc="upper left")
    plt.grid(True)
    if (exportPlotInPNG):
        plt.savefig(exportPath+exportName+".png")
    else:
        plt.show()

# teX and teY are the test set
# activations= is a theano function with activation outputs of every layer
#   activations = theano.function(inputs=[X], outputs=[l1, l2, l3, l4, py_x], allow_input_downcast=True)
def plotLayerStadistics(teX, activations=None, exportName=["Conv1","Conv2","Conv3","FC1","FC2"], exportPath=""):

    def selectionSort(max,min,average,std):
        tam = len(max)
        for i in range(0,tam-1):
            MAX=i
            for j in range(i+1,tam):
                if max[MAX] < max[j]:
                    MAX=j

            a1,a2,a3,a4 = max[MAX],min[MAX],average[MAX],std[MAX]
            max[MAX],min[MAX],average[MAX],std[MAX]= max[i],min[i],average[i],std[i]
            max[i],min[i],average[i],std[i] = a1,a2,a3,a4


    w1=[]; w2=[]; w3=[]; w4=[]; w5=[]
    values = [w1, w2, w3, w4, w5]
    for start, end in zip(range(0, len(teX), 1), range(1, len(teX), 1)):
        x = activations(teX[start:end])
        x = [x[j][0] for j in range(len(x))]
        for j in range(0,len(x),1):
            if (len(x[j].shape)>2):
                x[j] = [np.reshape(x[j][i], (1,np.product(x[j][i].shape)))[0] for i in range(x[j].shape[0])]
            else:
                x[j] = np.reshape(x[j], (1,np.product(x[j].shape)))[0]
            values[j].append(x[j])
    values = [np.array(values[j]) for j in range(len(values))]
    exportPath = exportPath + "LayerStadistics/"
    if os.path.exists(exportPath)==0: os.mkdir(exportPath)
    for j in range(0,len(x),1):
        if (len(values[j].shape)>2):
            max = [ np.max(values[j][:,i,:]) for i in range(values[j].shape[1]) ]
            min = [ np.min(values[j][:,i,:]) for i in range(values[j].shape[1]) ]
            average = [ np.sum(values[j][:,i,:])/(values[j].shape[0]*values[j].shape[2]) for i in range(values[j].shape[1]) ]
            std = [ np.std(values[j][:,i,:]) for i in range(values[j].shape[1]) ]
        else:
            max = [ np.max(values[j][:,i]) for i in range(values[j].shape[1]) ]
            min = [ np.min(values[j][:,i]) for i in range(values[j].shape[1]) ]
            average = [ np.sum(values[j][:,i])/(values[j].shape[0]) for i in range(values[j].shape[1]) ]
            std = [ np.std(values[j][:,i]) for i in range(values[j].shape[1]) ]
        selectionSort(max,min,average,std)
        plt.figure(num=None, figsize=(20, 15), dpi=60, facecolor='w', edgecolor='k')
        plt.plot(max,'b-',linewidth=2.0, label='Max')
        plt.plot(min,'r-',linewidth=2.0, label='Min')
        plt.plot(average,'g-',linewidth=2.0, label='Average')
        plt.errorbar(0, average[0], yerr=std[0], linewidth=1.0, color='yellow',label='Std')
        for i in range(1,len(average)):
            plt.errorbar(i, average[i], yerr=std[i], linewidth=1.0, color='yellow')
        plt.legend(loc="upper right")
        plt.savefig(exportPath+exportName[j]+".png")
