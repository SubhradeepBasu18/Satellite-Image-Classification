# Satellite Image Classification using Convolutional Neural Networks (CNN)

This project is focused on classifying satellite images into various categories, such as cloudy, desert, green area, and water, using Convolutional Neural Networks (CNN). The dataset for this project is sourced from Kaggle and contains satellite images with different geographical features.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Model Architecture](#model-architecture)

## Project Overview

The goal of this project is to build a deep learning model to classify satellite images into different categories. We utilize a Convolutional Neural Network (CNN) for image processing and classification. The model is trained on a Kaggle satellite image dataset, and its performance is evaluated based on classification accuracy.

### Categories in the Dataset:
- **Cloudy**: Satellite images showing cloudy areas.
- **Desert**: Images of desert landscapes.
- **Green Area**: Images of green areas such as forests or fields.
- **Water**: Satellite images showing bodies of water.

## Dataset

The dataset used in this project is a collection of satellite images categorized into four classes: **cloudy**, **desert**, **green_area**, and **water**. The images are available on Kaggle and can be downloaded [here](https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification).

The images are then preprocessed and resized before being fed into the CNN model.

## Requirements

To run this project, you need to have the following libraries installed:

- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Pillow
- scikit-learn

## Model Architecture

The model is based on a Convolutional Neural Network (CNN) and is structured as follows:

1. **Input Layer**: Takes images resized to (180, 180, 3) as input.
2. **Convolutional Layers**: Multiple convolutional layers to extract spatial features from the images.
3. **Max-Pooling Layers**: To reduce the spatial dimensions of the feature maps.
4. **Fully Connected Layers**: Dense layers to classify the features extracted by the convolutional layers.
5. **Output Layer**: Softmax activation function to output probabilities for each class.

## Results

The model is evaluated on a separate test set and achieves an accuracy of approximately **90%**
