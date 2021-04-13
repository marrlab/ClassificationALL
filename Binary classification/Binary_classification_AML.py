import numpy as np
import os
import glob
import random
import pickle
from skimage import io
from DataPreperationBinary import DataPrep
from sklearn import metrics
from keras import backend as K
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint


# PARAMETERS ##################################################################

path_to_train_images = "" # Change to same directory where the training images are placed by DataPrep
path_to_save = ""         # Change to directory where results will be placed 
                          # Directoy must contain "models", "raw_data", and "images" folders
                          # where the "images" folder contains a "test" and "train" folder
                          # and the "models" folder contains folder for trainingsizes (i.e. 10, 20, ... , 200)
path_to_test_images = ""  # Change to same directory where the testing images are placed by DataPrep

epoch_nr = 500  # Number of epochs for training 
early_stop = 50 # Number of non-decreasing losses to consider before stopping training
ntest = 600     # (even) Number of images per test fold
ntrain_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200] # What (even) trainingsizes to consider


# LOADING DATA ################################################################

folds = 10             # The implemented cross validation is hard coded with K=10 (wrt datapreperation)
nmax = ntrain_list[-1] # Maximum number of images necessary for training
DataPrep(nmax, ntest)  # Load train and test images
labelsTrainAll = [0]*int(nmax/2) + [1]*int(nmax/2) # ntrain_max 50/50 images
labelsTest = [0]*int(ntest/2) + [1]*int(ntest/2)   # ntest 50/50 images (per fold)

pathIm, dirsIm, filesIm = next(os.walk(path_to_train_images))
imagesTrainAll = []    
for i in range(len(filesIm)):
    im = io.imread(pathIm + filesIm[i],0) # Add images to array
    im = im[:,:,0:3]
    imagesTrainAll.append(im)    
imagesTrainAll = np.stack(imagesTrainAll,axis=0)


# CNN ARCHITECTURE ############################################################

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
        model.add(layers.Dense(1, activation='sigmoid'))        

        return model

# EXPERIMENT ##################################################################

for ntrain in ntrain_list:

    files = glob.glob(path_to_train_images+'*')
    for f in files: # Reset for next number of training images
        os.remove(f)
        
    img_width, img_height = 400, 400 # Size op input images
    if K.image_data_format() == 'channels_first':
            input_shape = (3, img_width, img_height)
    else:
            input_shape = (img_width, img_height, 3)
             
    
    # Generators (no augmentation)
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_datagen = ImageDataGenerator(rescale=1./255)
    
    
    # For saving results
    which_im_test = []  # Names for test images
    which_im_train = [] # Names for train images
    AUC_res = {}        # AUC results per fold

    # Incrementel addition (images are already in random order so range(ntrain[-1]) is allowed)
    # Select half non-blast (first half) and half blast (second half)
    idxTrain = list(range(nmax))[:int(ntrain/2)] + list(range(nmax))[int(nmax/2):int(nmax/2)+int(ntrain/2)]
    imagesTrain = imagesTrainAll[idxTrain,:,:]
    labelsTrain = labelsTrainAll[idxTrain]
    
    which_im_train.append([filesIm[i].split('.')[1] for i in idxTrain])
        
    for CV in range(folds):
        
        model = setup_sequential_model()
        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        
        idxTest = list(range(len(filesIm)))
        # Load testing images from correct fold
        path_to_test_images_fold = path_to_test_images + "/fold" + str(CV+1) + "/"

        pathIm, dirsIm, filesIm = next(os.walk(path_to_test_images_fold))
        imagesTest = []    
        for i in range(len(filesIm)):
            im = io.imread(pathIm + filesIm[i],0) # Add images to array
            im = im[:,:,0:3]
            imagesTest.append(im)
            
        imagesTest = np.stack(imagesTest,axis=0)
                    
        which_im_test.append([filesIm[i].split('.')[1] for i in idxTest])

        
        # Train network
        BatchSize = 5
        train_generator = train_datagen.flow(
                imagesTrain, labelsTrain,
                batch_size=BatchSize,
                shuffle=False)
        
        random_idx = random.sample(list(range(ntest)), int(ntest/4)) # Validation images
        
        val_generator = test_datagen.flow(
                imagesTest[random_idx,:,:,:], [labelsTest[i] for i in random_idx],
                batch_size=BatchSize,
                shuffle=False)
        
        checkpoint = ModelCheckpoint(path_to_save + "models/" + str(ntrain) + "/model_" + str(CV+1) + '_checkpoint' + '.hdf5', 
                                     monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=early_stop, verbose=0)
         
        history = model.fit_generator(
                    train_generator,
                    steps_per_epoch= int(np.ceil(len(labelsTrain)/BatchSize)),
                    epochs=epoch_nr,
                    validation_data=val_generator,
                    validation_steps= int(np.ceil(len(labelsTest)/BatchSize)),
                    callbacks = [checkpoint, early_stopping],
                    verbose=0)
        
        model.save(path_to_save + "models/" + str(ntrain) + "/model_" + str(CV+1) + '.h5')
        val_acc = history.history['val_accuracy']


        # Testing network
        test_generator = test_datagen.flow(
                    imagesTest, labelsTest,
                    batch_size=1,
                    shuffle=False)    
            
        predictions = model.predict_generator(
                test_generator, 
                steps=len(labelsTest),
                verbose=0)
    
        y_true = labelsTest
        y_pred = np.array(predictions)
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred) 
        AUC = metrics.auc(fpr, tpr)
        AUC_res['fold {}'.format(CV+1)] = AUC
        
        
        # Save variables
        with open(path_to_save + "raw_data/" + str(ntrain) + '.pkl', 'wb') as f:
            pickle.dump(AUC_res, f)
        
        with open(path_to_save + "images/train/" + str(ntrain) + '.pkl', 'wb') as f2:
            pickle.dump(which_im_train, f2)
        
        with open(path_to_save + "images/test/" + str(ntrain) + '.pkl', 'wb') as f3:
            pickle.dump(which_im_test, f3)
        
        
        # Monitor during training
        print(str(ntrain) + " fold: " + str(CV+1) + " - val acc: " + str(max(val_acc)) + " - AUC: " + str(AUC))
