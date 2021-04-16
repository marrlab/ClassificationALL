### IMPORT PACKAGES
import numpy as np
import math 
import random
from keras.preprocessing.image import ImageDataGenerator
img_gen = ImageDataGenerator()


def augmentation(images, labels, nrofimagesout):
    
    ### DEFINE AUGMENTATION POSSIBILITIES
    aug_datagen = ImageDataGenerator(
        rotation_range=359,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')


    ### FIND ALL UNIQUE LABELS IN DATASET
    uniqueLabels = np.unique(labels)
    
    ## Define new label values that range from 0 to the number of unique labels 
    newlabelvalues = list(range(len(uniqueLabels)))
    
    ### AUGMENT IMAGES
    ## Define empty arrays for the new images and labels
    newlabelsTrain = []
    newimagesTrain = []
    
    ### FOR EACH UNIQUE LABEL
    for i in range(len(uniqueLabels)):
        
        ### FIND THE INDICES OF THIS LABEL
        imageIndices, = np.where(labels == uniqueLabels[i])
        imageIndices = list(imageIndices)
        
        
        ### CALCULATE THE NUMBER OF AUGMENTED IMAGES PER ORIGINAL IMAGE
        perImage = int(math.ceil(nrofimagesout/len(imageIndices)))    
        
        
        ### SHUFFLE IMAGE INDICES
        random.shuffle(imageIndices)
        count = len(imageIndices)
        
        
        ### FOR EACH ORIGINAL IMAGE
        for j in imageIndices:
            currentImage = images[j]
            
            ### ADD ORIGINAL IMAGE AND LABEL TO THE NEW ARRAYS
            newlabelsTrain.append(newlabelvalues[i])
            newimagesTrain.append(currentImage[:,:,0:3])
                 
            
            ### CREATE X NUMBER OF AUGMENTED IMAGES
            for k in range(perImage):
                                
                ### AS LONG AS THE NUMBER OF IMAGES IS NOT LARGER THAN THE TOTAL
                if count < nrofimagesout:
                    
                    ### CREATE NEW AUGMENTED IMAGE
                    ImageNew = aug_datagen.random_transform(currentImage)
                    
                    
                    ### SAVE NEW IMAGE AND LABEL TO THE ARRAYS
                    newlabelsTrain.append(newlabelvalues[i])
                    newimagesTrain.append(ImageNew[:,:,0:3])
                    
                    
                    ### COUNT NUMBER OF IMAGES IN THE ARRAY
                    count = count + 1
    
    ### CHANGE SHAPE OF IMAGE ARRAY
    newimagesTrain = np.stack(newimagesTrain,axis=0)  


    ### RETURN BOTH ARRAYS
    return newimagesTrain, newlabelsTrain
