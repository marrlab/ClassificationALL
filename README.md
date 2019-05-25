# ClassificationALL
These scripts were used in the article titled "Accurate classification of blood cells in a small acute lymphoblastic leukemia dataset using convolutional neural networks" by Schouten et al. The networks were trained on a NVIDIA GeForce GT 730 GPU with Python version 3.5, keras version 2.0.8 and tensorflow-GPU version 1.4.0.

## Dataset
The dataset was obtained by filling in the application form on this site: https://homes.di.unimi.it/scotti/all/. The dataset consists of two parts. ALL-IDB1 contains the larger blood smear images. ALL-IDB2 contains the single-cell images that are extracted from the larger ALL-IDB1 images. More information can be found in our article or in the article of the owners of the dataset [1]. 

## Scripts
In both the binary classification as multiclass classification scripts, the user input should be filled in to be able to train the networks. This includes paths to the images and labels folder and the folder to save the results. 

### Binary classification
#### Binary_classification.py
In the binary classification script, networks can be trained with 10-fold cross-validation and different training set sizes as explained in the article. During training, the accuracies and losses are saved in an excel sheet and the weights of the network that has the lowest validation loss are saved. After training, the weights of the best network are loaded and the images in the test set are tested to evaluate the classification performance. Along with the predictions, the sensitivities and specificities are calculated to be able to create a curve and calculate the area under the curve (AUC).

#### Saliency.py
Here, saliency maps can be created to visualize the focus areas of the network when it tries to classify an image. This is done by loading a network with the pretrained weights and the testing images. These images are then classified by the network and at the same time the visualisation is made. These images are then saved in the assigned folder. 

### Multiclass classification
#### Multiclass_classification.py
In the multiclass classification script, the classification of six classes in the ALL dataset can be trained with 10-fold cross-validation and a training set containing 200 images. 

#### Multiclass_data_augmentation.py

## References
[1] R. D. Labati, V. Piuri, and F. Scotti, “All-idb: The acute lymphoblastic leukemia
image database for image processing,” in Image processing (ICIP), 2011 18th IEEE
international conference on, pp. 2045–2048, IEEE, 2011.
