### IMPORT PACKAGES
import numpy as np
from datetime import datetime
import os
import random
import pandas as pd
from skimage import io
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras import layers, models
import pickle
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.metrics import confusion_matrix


### USER INPUT
data_augmentation = 0               #Data augmentation yes(1) or no(0)
path_to_images = ""
path_to_labels = ""
path_to_save = ""
modelname = ""
epoch_nr = 500
folds = range(10)
img_width, img_height = 257, 257

## Sizes of test and validation set
sizeTest = 25
sizeVal = 25

## Ratio between blast and non-blast images in the test set 
sizes0 = [12,13,12,13,12,13,12,13,12,12]
sizes1 = [13,12,13,12,13,12,13,12,13,13]

## Sizes of training set
sizesTrain = range(10, 201,10)


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
        model.add(layers.Dense(1, activation='sigmoid', name='preds'))

        return model

if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
else:
        input_shape = (img_width, img_height, 3)


### GENERATORS
test_datagen = ImageDataGenerator(rescale=1./255)
if data_augmentation == 1:
    train_datagen = ImageDataGenerator(
            rotation_range=359,
            horizontal_flip=True,
            vertical_flip=True,
            rescale=1./255,    
            fill_mode='nearest')        
elif data_augmentation == 0:
    train_datagen = ImageDataGenerator(rescale=1./255)


### IMPORT IMAGES
pathIm, dirsIm, filesIm = next(os.walk(path_to_images))
        
imagesAll = []    
imageNrs = []
for i in range(len(filesIm)):
    # Add images to array
    im = io.imread(pathIm + filesIm[i],0) 
    im = im[:,:,0:3]
    imagesAll.append(im)
    imageNrs.append(int(filesIm[i][2:5])-1)
    
imagesAll = np.stack(imagesAll,axis=0)  


### IMPORT LABELS
df = pd.read_excel(path_to_labels + 'Supplementary Table S1.xlsx')

labelsAll = df[['Binary class']].values.flatten()
labelsAll = labelsAll[imageNrs]
labelsAll = labelsAll.astype(int).reshape(labelsAll.size)

## Randomize indices to randomize images in test set
indices0, = np.where(labelsAll == 0)
indices1, = np.where(labelsAll == 1)

random.shuffle(indices0)
random.shuffle(indices1)


### RUN FOR EACH FOLD
for CV in folds:
    ### DEFINE IMAGES AS TESTING AND VALIDATION IMAGES
    idxTest = np.concatenate((indices0[sum(sizes0[:CV]):sum(sizes0[:CV])+sizes0[CV]],
                                            indices1[sum(sizes1[:CV]):(CV+1)*sizes1[CV]]), axis=0)
    
    if CV == folds[-1]:
        idxVal = np.concatenate((indices0[0:sizes0[0]], indices1[0:sizes1[0]]), axis=0)
    else:
        idxVal = np.concatenate((indices0[sum(sizes0[0:CV+1]):sum(sizes0[0:CV+1])+sizes0[CV+1]],
                                          indices1[sum(sizes1[0:CV+1]):sum(sizes1[0:CV+1])+sizes1[CV+1]]))
    
    ### DEFINE OTHER IMAGES AS TRAINING IMAGES
    idxTrain = [i for j, i in enumerate(list(range(len(labelsAll)))) if j not in idxVal and j not in idxTest]
    idxTrain = np.array(idxTrain, dtype='int64')
    
    indices0Train, = np.where(labelsAll[idxTrain]==0)
    indices1Train, = np.where(labelsAll[idxTrain]==1)
    
    indices0Train = idxTrain[indices0Train]
    indices1Train = idxTrain[indices1Train]
    
    ### SHUFFLE TRAINING IMAGES TO RANDOMIZE THE IMAGES THAT ARE IN THE TRAINING SET IN EACH STEP
    random.shuffle(indices0Train)
    random.shuffle(indices1Train)   

      
    ### START TRAINING NETWORKS WITH DIFFERENT SIZES OF TRAINING SETS
    for index in range(len(sizesTrain)): 
        
        ### LOAD NEW NETWORK
        model = setup_sequential_model()
        model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])       
            
        
        ### DEFINE NEW TRAINING SET WITH X NUMBER OF IMAGES
        if index == len(sizesTrain)-1:
            idxTrainNew0 = indices0Train
            idxTrainNew1 = indices1Train

            idxTrainNew = np.concatenate((idxTrainNew0, idxTrainNew1), axis=0)       
        else:
            idxTrainNew0 = indices0Train[range(0,(sizesTrain[index])//2)]
            idxTrainNew1 = indices1Train[range(0,(sizesTrain[index])//2)]

            idxTrainNew = np.concatenate((idxTrainNew0, idxTrainNew1), axis=0)
        
       
        ### DEFINE IMAGE AND LABEL SETS
        ## Training images and labels
        imagesTrain = imagesAll[idxTrainNew,:,:]
        labelsTrain = labelsAll[idxTrainNew]
        
        ## Validation images and labels
        imagesVal = imagesAll[idxVal,:,:]
        labelsVal = labelsAll[idxVal]
    
        
        ### TRAIN NETWORK
        ## Define generators
        train_generator = train_datagen.flow(
                imagesTrain, labelsTrain,
                batch_size=5,
                shuffle=True)
        
        val_generator = test_datagen.flow(
                imagesVal, labelsVal,
                batch_size=5,
                shuffle=True)
        
        ## Define callbacks
        checkpoint = ModelCheckpoint(path_to_save + modelname + '_' + str(sizesTrain[index]) + '_' + str(CV) + '_checkpoint' + '.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        csvlog = CSVLogger(path_to_save + modelname + '_' + str(sizesTrain[index]) + '_' + str(CV) + '_train_log.csv',append=False)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1)
        
        ## Start training
        startTime = datetime.now()    
        history = model.fit_generator(
                    train_generator,
                    steps_per_epoch=len(idxTrainNew)//5,
                    epochs=epoch_nr,
                    validation_data=val_generator,
                    validation_steps=len(idxVal)//5,
                    callbacks = [checkpoint, csvlog, early_stopping])
        
        Time = datetime.now() - startTime

       
        ### SAVE DATA
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
                
            
        ### TESTING NETWORK
        ## Loading weights of best network that was just trained
        model = setup_sequential_model()
        model = load_model(path_to_save + modelname + '_' + str(sizesTrain[index]) + '_' + str(CV) + '_checkpoint' + '.hdf5')
        model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    
    
        ### DEFINE TESTING IMAGES AND LABELS
        imagesTest = imagesAll[idxTest,:,:]
        labelsTest = labelsAll[idxTest]


        ### DEFINE TEST GENERATOR
        test_generator = test_datagen.flow(
                    imagesTest, labelsTest,
                    batch_size=1,
                    shuffle=False)    
          
        
        ### TESTING NETWORK 
        ## Get predictions for network
        predictions = model.predict_generator(
                test_generator, 
                steps=len(labelsTest),
                verbose=1)        
    
        ## Calculate sensitivities and specificities
        sens = []
        spec = []
        for j in range(1000):
            predNew = np.zeros((len(predictions), 1))
            predNew[predictions > j/999] = 1
            
            tn, fp, fn, tp = confusion_matrix(labelsTest, predNew).ravel()        
            sens.append(tp/(tp+fn))
            spec.append(tn/(tn+fp))
        
        ## Calculate area under the curve
        auc = np.trapz(sens,spec)
        
        ### SAVE VARIABLES
        with open(path_to_save + modelname + '_' + str(sizesTrain[index]) + '_' + str(CV) + '.pkl', 'wb') as f:
            pickle.dump([acc, val_acc, loss, val_loss, Time, predictions, sens, spec, auc, idxTest, idxVal, idxTrainNew, indices0, indices1, indices0Train, indices1Train], f)


          
