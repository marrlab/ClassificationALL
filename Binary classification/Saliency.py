from vis.visualization import visualize_saliency
from vis.utils import utils
from keras import activations
from matplotlib import pyplot as plt
import pickle
import numpy as np
from skimage import io
import os
import matplotlib
from keras import backend as K
from keras.models import load_model
from keras import layers, models


### USER INPUT
PATH_TO_IMAGES = ""
PATH_TO_SAVE = ""
MODELNAME = ""


sizesTrain = range(10, 201,10)
folds = range(10)

   
### IMPORT IMAGES
pathIm, dirsIm, filesIm = next(os.walk(PATH_TO_IMAGES))
        
imagesAll = []    
imageNrs = []
for i in range(len(filesIm)):
    # Add images to array
    im = io.imread(pathIm + filesIm[i],0) 
    im = im[:,:,0:3]
    imagesAll.append(im)
    imageNrs.append(int(filesIm[i][2:5]))
    
imagesAll = np.stack(imagesAll,axis=0)  

       
### RUN OVER ALL FOLDS
for CV in folds:
    
    ### OPEN PICKLE TO GET TEST IMAGE INDICES
    f = open(PATH_TO_SAVE + MODELNAME + '_' + str(10) + '_' + str(CV) + '.pkl', 'rb')
    obj = pickle.load(f)
    f.close()
    
    ## Extract test image indices
    idxTest = obj[7]  
    
    ## Define test images
    imagesTest = imagesAll[idxTest,:,:,:]    
        
    
    ### RUN OVER TRAINING SET SIZES TO COMPARE NETWORKS
    for j in range(len(sizesTrain)):     
        
        ### CREATE NETWORK AND LOAD WEIGHTS
        model = load_model(PATH_TO_SAVE + MODELNAME + '_' + str(sizesTrain[j]) + '_' + str(CV) + '_checkpoint' + '.hdf5')
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])
            
        ### GET LAYERNAMES
        layernames = []
        for layer in model.layers:
            layernames.append(layer.name)    
        
        
        ### RUN OVER ALL TEST IMAGES
        for i in range(len(idxTest)): 
            
            ### FIND INDEX FOR LAST LAYER (DENSE LAYER)
            layer_idx = utils.find_layer_idx(model, layernames[-1])
            
            
            ### CHANGE ACTIVATION
            model.layers[layer_idx].activation = activations.linear
            model = utils.apply_modifications(model)
            
            ### GET SALIENCY MAP
            grads = visualize_saliency(model, layer_idx, filter_indices=0, seed_input=imagesTest[i])
            
            ## Plot saliency map
            plt.imshow(grads, cmap='jet')
            
            ## Save saliency map            
            matplotlib.image.imsave(PATH_TO_SAVE + 'Images/' + str(CV) + '/' + MODELNAME + '_' + str(sizesTrain[j]) + '_' + str(CV) + '_' + str(imageNrs[idxTest[i]]) + '.jpg', grads)
