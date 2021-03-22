from keras import activations
from matplotlib import pyplot as plt
import pickle
import numpy as np
from skimage import io
import os
import pandas as pd

from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D
from keras import layers, models
import tensorflow as tf
from tensorflow import keras

from keras.preprocessing.image import ImageDataGenerator

import imp
import innvestigate
import innvestigate.utils as iutils
# Use utility libraries to focus on relevant iNNvestigate routines.
eutils = imp.load_source("utils", "")
mnistutils = imp.load_source("utils_mnist", "")


### USER INPUT
path_to_images = ""
path_to_save = ""
path_to_save_images = ""
path_to_labels = ""
modelname = ""
sizesTrain = list(range(10, 101,10)) + [150] + [200]
folds = range(10)
lrpmethod = 'LRPE'
#index = range(25)
    
### NETWORK STRUCTURE
def setup_sequential_model():
        model = models.Sequential()
        
        model.add(layers.Conv2D(4, (3, 3), activation='relu', input_shape=input_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(8, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(8, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(8, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(16, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(16, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
      
        model.add(layers.Flatten())
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(2, activation='softmax', name='preds'))

        return model

img_width, img_height = 257, 257
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
        input_shape = (img_width, img_height, 3)
   
### Import images
pathIm, dirsIm, filesIm = next(os.walk(path_to_images))
        
imagesAll = []    
imageNrs = []
for i in range(len(filesIm)):
    # Add images to array
    im = io.imread(pathIm + filesIm[i],0) 
    im = im[:,:,0:3]
    imagesAll.append(im)
    imageNrs.append(int(filesIm[i][2:5]))
    
imagesAll = np.stack(imagesAll,axis=0)  

### Import labels
df = pd.read_excel(path_to_labels + 'labels.xls')

labelsAll = df[['Binary']].values.flatten()
labelsAll = labelsAll[imageNrs]

test_datagen = ImageDataGenerator(rescale=1./255)

def input_postprocessing(X):
    return X / 255

# Configure analysis methods and properties
if lrpmethod == "LRPZ":
    methods = [
        # NAME                    OPT.PARAMS                POSTPROC FXN               TITLE
        ("input",                 {},                       input_postprocessing,      "Input"),
        ("lrp.z",                 {},                       mnistutils.heatmap,        "LRP-Z"),
        ]
    
elif lrpmethod == "LRPE":
     methods = [
        # NAME                    OPT.PARAMS                POSTPROC FXN               TITLE
        ("input",                 {},                       input_postprocessing,      "Input"),
        ("lrp.epsilon",           {"epsilon": 1},           mnistutils.heatmap,        "LRP-Epsilon"),
        ]



for CV in folds:
    ## Open pickle
    f = open(path_to_save + modelname + '_' + str(10) + '_' + str(CV) + '.pkl', 'rb')
    obj = pickle.load(f)
    f.close()
    
    idxTest = obj[7]  
    
    imagesTest2 = imagesAll[idxTest,:,:,:]    
    labelsTest2 = labelsAll[idxTest]       
    
    for j in range(len(sizesTrain)):     
        #model = setup_sequential_model()
        model = load_model(path_to_save + modelname + '_' + str(sizesTrain[j]) + '_' + str(CV) + '_checkpoint' + '.hdf5')
        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])
        
        
        ## Get layernames
        layernames = []
        for layer in model.layers:
            layernames.append(layer.name) 
            
        
        # Create model without trailing softmax
        model_wo_softmax = iutils.keras.graph.model_wo_softmax(model)
        
        # Create analyzers.
        analyzers = []
        for method in methods:
            analyzer = innvestigate.create_analyzer(method[0],        # analysis method identifier
                                                    model_wo_softmax, # model without softmax output
                                                    **method[1])      # optional analysis parameters
        
            # Some analyzers require training.
            # analyzer.fit(data[0], batch_size=256, verbose=1)
            analyzers.append(analyzer)
        
        
        test_images = list(zip(imagesTest2, labelsTest2))
        analysis = np.zeros([len(test_images), len(analyzers), 257, 257, 3])
        text = []
        
        label_to_class_name = ['0', '1']
        
        
        for i, (x, y) in enumerate(test_images):
            # Add batch axis.
            x = x[None, :, :, :]
           
            
            # Predict final activations, probabilites, and label.
            presm = model_wo_softmax.predict_on_batch(x)[0]
            prob = model.predict_on_batch(x)[0]
            y_hat = prob.argmax()
            
            # Save prediction info:
            text.append(("%s" % label_to_class_name[y],    # ground truth label
                         "%.2f" % presm.max(),             # pre-softmax logits
                         "%.2f" % prob.max(),              # probabilistic softmax output  
                         "%s" % label_to_class_name[y_hat] # predicted label
                        ))
        
            for aidx, analyzer in enumerate(analyzers):
                # Analyze.
                a = analyzer.analyze(x)
                
                # Apply common postprocessing, e.g., re-ordering the channels for plotting.
                a = mnistutils.postprocess(a)
                # Apply analysis postprocessing, e.g., creating a heatmap.
                a = methods[aidx][2](a)
                # Store the analysis.
                analysis[i, aidx] = a[0]
        
        for i in range(len(idxTest)):
            fig, ax = plt.subplots(figsize=plt.figaspect(analysis[i,0]))
            fig.subplots_adjust(0,0,1,1)
            ax.imshow(analysis[i,0])
            plt.savefig(path_to_save_images + "{}_CV{}_Img{}.jpg".format(lrpmethod, str(CV), str(imageNrs[idxTest[i]])))
            plt.close()
            
            
            fig, ax = plt.subplots(figsize=plt.figaspect(analysis[i,1]))
            fig.subplots_adjust(0,0,1,1)
            ax.imshow(analysis[i,1])
            plt.savefig(path_to_save_images + "{}_CV{}_Img{}_W_TrainSize{}.jpg".format(lrpmethod, str(CV), str(imageNrs[idxTest[i]]), str(sizesTrain[j])))
            plt.close()
            
