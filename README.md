# CNN-Based Segmentation Models for Lung CT Scans

# Introduction:

In this project, we develop and compare multiple CNN-based segmentation models for lung CT scans.

Computed tomography (CT) is an imaging procedure that uses different  penetrability of X-ray on different organizations of body to exploit targeted regions. A patient lies on a table which slides into a hole around an annulus. A CT scanner emits a series of narrow X-ray beams through the human body from the X-ray source. These beams are captured by the detector at the opposite end of the machine. A series of scans are taken at different angles as the gantry rotates. Each scan produces a 2D slice of the body; the slices can then be used to construct complex 3D structure of the scanned body by a back projection algorithm. CT scans are used by physicians and radiologists to provide patient diagnosis for a wide range of diseases. Subtle changes in size, shapes, or structure are promising key factors to confirm or discard the presence of certain diseases or infections.  As the demand for CT scan data increases, there is a vastly growing field for computer-aided diagnosis, particularly automated systems and algorithms, to assist with clinical diagnosis and treatment plans. One of these methods is deep-learning-based instance segmentation. A common method is to train a specifically designed convolutional neural network that can assign a label for each pixel on an image to distinguish which part of the image belongs to some certain types of objects or structures that we are interested in.


# Methodology: 

For this project, we are interested in labeling infected areas in whole lung scans of COVID-19 patients. Medical imaging with computed tomography plays an essential role in diagnosing COVID-19. Radiologists have identified irregularities in computed tomography scans of the lungs of COVID-19 patients. These irregularities can be quantified through semantically segmenting medical lung images of COVID-19 patients. In general, given an image, our goal is to accurately predict whether a part of the image is lung and whether it is infected or not. The dataset we use is a 2020 COVID-19 CT scan dataset that is publicly accessible on the Kaggle database (https://www.kaggle.com/datasets/luisblanche/covidct). This dataset contains CT scans of the lungs of COVID-19 patients with pixel-wise annotation for the healthy and infected regions on the lung. We have 3 different classes for each image: the lung, the infected, and the background. For our experiment, we split the data into a training, validation and testing set, according to the proportion: 65%, 13% and 22%, which is a generally accepted splitting threshold in the machine learning community. We implement and compare several state-of-the-art deep learning networks in segmentation.

The three segmentation models we are comparing are the U-Net, fully convolutional network (FCN) and the Pyramid Scene Parsing Network (PSPNet). 

In general, semantic segmentation convolutional neural network models contain an encoder and a decoder. The encoder extracts important features from the input image and the decoder predicts the classification of the pixels at the end of the model. 

The U-Net (UNet) model can be visualized with a U shaped architecture in which the left side of the model is the encoder and the right hand side of the model is the decoder. The encoder, or the contractor path, contains a convolutional and max pooling downsampling layer, which extracts important features from the image and converts the input image into feature representations. The decoder, or the expansion path, contains the transposed convolutional layers that can expand the image using localization. The decoder uses the features learned by the encoder, upsampling, and concatenation to project discriminative features onto the pixel space for dense classification.

The Fully Convolutional Network (FCN) uses the height and width of intermediate feature maps to reconstruct the input image with a transposed convolutional layer. It uses the convolution neural network architecture to extract important image features and converts the number of channels into classes using a 1x1 convolutional layer. This layer is used to perform a transposed convolution to transform the height and width of the feature maps into the input image. 

The Pyramid Scene Parsing Network (PSPNet) considers the global context of the image to make local level predictions. It uses dilated convolutions instead of traditional convolutional layers to increase the receptive field for richer features. This model also uses a pyramid pooling module, in which the feature map from the backbone is pooled at different sizes and passed through a convolutional layer for upsampling. The upsampled layers are then concatenated with the original feature map. This allows for the fusion of features at different scales and for the decoding step to be dependent on the aggregation of the overall context of the image.

Before training, we performed a dataset screening. This is because at both ends of the scan, there are slices that contains neither lung or infected areas (only the background), which significantly increases the number of background pixels for the data and cause an unbalanced data distribution. To prevent any biases to the model, we picked the middle third of each scan to ensure every slice contains all lung, infection, and background.

In terms of training, we implemented a binary cross-entropy loss and an Adam Optimizer. For the hyper parameters, we set the learning rate (LR) to 1e-4, the weight decay (wd) to 1e-5, and the epoch number to 40.


To evaluate the prediction, we quantify the data by calculating the pixel-wise accuracy and the intersection over union, usually known as the IoU. The pixel-wise accuracy reports the number of correctly classified pixels divided by the number of pixels of the whole image. For tasks with imbalanced number of pixels between different classes or huge background area, this metric is not quite useful. A commonly used metric in segmentation to address an imbalanced number of pixels is the IoU. For each class, the IoU is calculated by dividing the intersected area between prediction and ground truth by the union area. The IoU is used to evaluate how accurately the prediction can locate each area.


# Results and Discussion: 

By comparing the accuracy of the predictions and the mean intersection of union (MIoU), we found that the PSPNet model does a better job in terms of predicting lung and background; however, what we primarily concern is the accuracy of predicting the infected areas. The FCN model performed better than the other models. 

We also attempt to improve the result by using some image preprocessing techniques such as histogram equalization and data augmentation. We also try a different loss function called the focal loss. The focal loss is designed to handle imbalanced number of pixels between different classes in an image. We can see from the results that, although these attempts are not very successful, they are still interesting explorations for our project.

