# image_processing_with_animals

This project focuses on classifying images of animals using a Convolutional Neural Network (CNN). The goal is to train a model that can accurately identify different animal species based on the given dataset of images.

## Dataset

The dataset consists of labeled images of various animals, organized into categories. Each image is resized to 128x128 pixels for consistency. The dataset is split into training and testing sets, with the goal of training the model on the training set and evaluating its performance on the testing set.
You can access the dataset here:
[Animals with Attributes 2 Dataset on Kaggle](https://www.kaggle.com/datasets/rrebirrth/animals-with-attributes-2)


### Dataset Features:
- **Image Dimensions**: 128x128 pixels
- **Number of Classes**: Multiple animal species (specific classes are provided in the dataset)
- **Data Augmentation**: To increase model robustness, data augmentation techniques like rotation, shifting, and zooming are applied to the training set.

## Technologies and Libraries

The project utilizes the following libraries and tools:

- **TensorFlow**: For building and training the deep learning model.
- **Keras**: High-level neural networks API, running on top of TensorFlow.
- **OpenCV**: For image manipulation tasks such as contrast and brightness adjustment, and color constancy algorithms.
- **Matplotlib**: For visualizing training and validation accuracy and loss.
- **NumPy**: For numerical operations and data handling.
  
### Data Augmentation:
Data augmentation techniques are used to improve the model's ability to generalize to unseen data:
- **Rotation**: Images are rotated randomly within a certain degree range.
- **Width and Height Shifting**: Randomly shifts the image horizontally or vertically.
- **Zooming and Shearing**: Zooming in or applying shear transformations to the image.
- **Horizontal Flipping**: Randomly flips images horizontally.

### Color Constancy:
The model also incorporates a color constancy algorithm to correct lighting issues in images, applying the **Gray World Algorithm** to improve image consistency.

## Model Architecture

The model is a Convolutional Neural Network (CNN) built with the following layers:

1. **Convolutional Layers**: Four convolutional layers with increasing filters (32, 64, 128, 256).
2. **Batch Normalization**: Applied after each convolutional layer to stabilize and speed up training.
3. **MaxPooling Layers**: Used to downsample the image after each convolutional layer.
4. **Fully Connected Layers**: Includes a dense layer with 512 units and a dropout layer to prevent overfitting.
5. **Softmax Output Layer**: For multi-class classification.

### Optimizer:
- **Adam Optimizer**: Used to minimize the sparse categorical crossentropy loss during training.

### Callbacks:
- **EarlyStopping**: Monitors the validation loss and stops training if there is no improvement for a set number of epochs.

## Project Link
You can find the complete analysis and code in the [Image Processing with Animals on Kaggle]([https://www.kaggle.com/code/remdilatkse/image-processing-with-animals]).


