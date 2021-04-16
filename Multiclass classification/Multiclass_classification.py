### IMPORT PACKAGES
import numpy as np
from datetime import datetime
import os
import pandas as pd
import pickle
from skimage import io
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras import layers, models, optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.metrics import f1_score

## Import data augmentation script
from Multiclass_data_augmentation import augmentation


DATA_AUGMENTATION = 1               #Data augmentation yes(1) or no(0)
PATH_TO_IMAGES = ""
PATH_TO_LABELS = ""
PATH_TO_CV = ""
PATH_TO_SAVE = ""
MODELNAME = ""
MAX_EPOCH = 500


folds = range(10)
output = 6
img_width, img_height = 257, 257


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
        model.add(layers.Dense(output, activation='softmax', name='preds'))

        return model

if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
else:
        input_shape = (img_width, img_height, 3)


## Generators
test_datagen = ImageDataGenerator(rescale=1./255)
if DATA_AUGMENTATION == 1:
    train_datagen = ImageDataGenerator(
            rotation_range=359,
            horizontal_flip=True,
            vertical_flip=True,
            rescale=1./255,    
            fill_mode='nearest')        
elif DATA_AUGMENTATION == 0:
    train_datagen = ImageDataGenerator(rescale=1./255)


### IMPORT IMAGES
pathIm, dirsIm, filesIm = next(os.walk(PATH_TO_IMAGES))
        
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
df = pd.read_excel(PATH_TO_LABELS + 'Supplementary Table S1.xlsx')

labelsAll = df[['Multiclass class']].values.flatten()
labelsAll = labelsAll[imageNrs]
labelsAll = labelsAll.astype(int).reshape(labelsAll.size)


### IMPORT CROSS-VALIDATION SPLITS
df = pd.read_excel(PATH_TO_CV + 'cvMultiple.xlsx', header=None)

cvIndices = df.values.flatten()


### RUN FOR EACH FOLD
for CV in folds:
    
    ### LOAD NEW NETWORK
    model = setup_sequential_model()
    adam = optimizers.Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['acc'])    
    
    ### DEFINE IMAGES AS TESTING AND VALIDATION IMAGES
    idxTest, = np.where(cvIndices == CV)
    
    if CV < 9:
        idxVal, = np.where(cvIndices == CV+1)
    else:
        idxVal, = np.where(cvIndices == 0)
     
        
    ### DEFINE OTHER IMAGES AS TRAINING IMAGES
    idxTrain = [i for j, i in enumerate(list(range(len(labelsAll)))) if j not in idxVal and j not in idxTest]
    idxTrain = np.array(idxTrain, dtype='int64')
    idxInTrain, = np.where(cvIndices[idxTrain] < 10)  
    idxTrain = idxTrain[idxInTrain]
    
    ## Define training images and labels
    imagesTrain = imagesAll[idxTrain,:,:,:]
    labelsTrain = labelsAll[idxTrain]
    
    ## Apply data augmentation on training images to fix imbalance
    newimagesTrain, newlabelsTrain = augmentation(imagesTrain, labelsTrain, 150)       
    
    ## Create one-hot encoding matrix with labels to put in network
    newlabelsTrainMatrix = np.zeros((len(newlabelsTrain), output))
    for i in range(len(newlabelsTrain)):
        newlabelsTrainMatrix[i, newlabelsTrain[i]] = 1
    
           
    ### DEFINE VALIDATION IMAGES AND LABELS
    imagesVal = imagesAll[idxVal,:,:,:]
    labelsVal = labelsAll[idxVal]

    ## Create one-hot encoding matrix for validation labels
    labelsValMatrix = np.zeros((len(labelsVal), output))  
    newlabelsVal = []
    for i in range(len(labelsVal)):
        idx, = np.where(np.unique(labelsTrain) == labelsVal[i])
        labelsValMatrix[i, int(idx)] = 1
        newlabelsVal.append(int(idx))    
      
        
    ### TRAIN NETWORK
    ## Define generators
    train_generator = train_datagen.flow(
            newimagesTrain, newlabelsTrainMatrix,
            batch_size=5,
            shuffle=True)
    
    val_generator = test_datagen.flow(
            imagesVal, labelsValMatrix,
            batch_size=5,
            shuffle=True)
    
    ## Define callbacks
    checkpoint = ModelCheckpoint(PATH_TO_SAVE + MODELNAME + '_' + str(CV) + '_checkpoint' + '.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    csvlog = CSVLogger(PATH_TO_SAVE + MODELNAME + '_' + str(CV) + '_train_log.csv',append=False)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1)
    
    ## Start training
    startTime = datetime.now()    
    history = model.fit_generator(
                train_generator,
                steps_per_epoch=len(newlabelsTrain)//5,
                epochs=MAX_EPOCH,
                validation_data=val_generator,
                validation_steps=len(labelsVal)//5,
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
    model = load_model(PATH_TO_SAVE + MODELNAME + '_' + str(CV) + '_checkpoint' + '.hdf5')
    adam = optimizers.Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy',
              optimizer=adam,
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
    
    ## Find label with highest probability
    uniqueLabelsTrain = np.unique(labelsTrain)
    test_pred_array = []
    for i in range(len(predictions)):
        indexMax = np.argmax(predictions[i,:])
        test_pred_array.append(uniqueLabelsTrain[indexMax])    
    
    ## Put ground truth next to network prediction in one array
    compare = np.column_stack((labelsTest,test_pred_array))
    
    ## Calculation of F1-score
    F1 = f1_score(compare[:,0], compare[:,1], average='weighted')

    
    ### SAVE VARIABLES
    with open(PATH_TO_SAVE + MODELNAME + '_' + str(CV) + '.pkl', 'wb') as f:
        pickle.dump([acc, val_acc, loss, val_loss, Time, predictions, compare, F1, idxTest, idxVal], f)
