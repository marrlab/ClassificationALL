# ClassificationALL
These scripts were used in the article titled "Tens of images can suffice to train neural networks for malignant leukocyte detection" by Schouten et al in Scientific Reports: https://www.nature.com/articles/s41598-021-86995-5#Sec13. The networks were trained on a NVIDIA GeForce GT 730 GPU with Python version 3.5, keras version 2.0.8 and tensorflow-GPU version 1.4.0.

## Dataset
The dataset was obtained by filling in the application form on this site: https://homes.di.unimi.it/scotti/all/. The dataset consists of two parts. ALL-IDB1 contains the larger blood smear images. ALL-IDB2 contains the single-cell images that are extracted from the larger ALL-IDB1 images. More information can be found in our article or in the article of the owners of the dataset [1]. 

## Scripts
In both the binary classification as multiclass classification scripts, the user input should be filled in to be able to train the networks. This includes paths to the images and labels folder and the folder to save the results. 

### Binary classification
#### Binary_classification.py
In the binary classification script, networks can be trained with 10-fold cross-validation and different training set sizes as explained in the article. During training, the accuracies and losses are saved in an excel sheet and the weights of the network that has the lowest validation loss are saved. After training, the weights of the best network are loaded and the images in the test set are tested to evaluate the classification performance. Along with the predictions, the sensitivities and specificities are calculated to be able to create a curve and calculate the area under the curve (AUC).

#### Saliency.py
Saliency maps can be created to visualize the focus areas of the network when it tries to classify an image. This is done by loading a network with the pretrained weights and the testing images. These images are then classified by the network and at the same time the visualization is made. This visualization is done using the Keras Visualization Toolkit that can be found here: https://github.com/raghakot/keras-vis. The augmented images are then saved to the assigned folder. 

### Multiclass classification
#### Multiclass_classification.py
In the multiclass classification script, the classification of six classes in the ALL dataset can be trained with 10-fold cross-validation and a training set containing 200 images. Because of the imbalance in the number of images per class, data augmentation is used to create 150 images for each class during training. The test images are again classified by the best network and the F1-score is calculated. 

#### Multiclass_data_augmentation.py
As mentioned before, 150 images are created using data augmentation for the multiclass classification due to the imbalance in the number of images per class. This data augmentation includes rotation and horizontal and vertical flipping, like in the binary classification. First it is calculated how many extra images should be created. Then the rotation and flipping is chosen randomly and the images are adjusted. The new image is then added to a new array of images. The same holds for the labels. These two arrays are then returned to the Multiclass_classicification.py script. 

## References
[1] R. D. Labati, V. Piuri, and F. Scotti, “All-idb: The acute lymphoblastic leukemia
image database for image processing,” in Image processing (ICIP), 2011 18th IEEE
international conference on, pp. 2045–2048, IEEE, 2011.
