# BirdCLEF 2024 - Bird Sound Classification

## Overview
This repository contains the code and methodology used for the BirdCLEF 2024 Kaggle competition, where I achieved a rank of 55th out of 974 participants, earning a bronze medal. The goal of this competition was to build a model that can accurately classify bird sounds.

## Table of Contents
- [Overview](#overview)
- [Data Loading & Preprocessing](#data-loading--preprocessing)
- [Data Augmentation](#data-augmentation)
- [Feature Engineering](#feature-engineering)
- [Model Building & Training](#model-building--training)
- [Inference](#inference)
- [Results](#results)
- [Conclusion](#conclusion)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Acknowledgments](#acknowledgments)

## Data Loading & Preprocessing
In this section, we load the bird sound datasets and preprocess them to prepare for training.

### Libraries and Dependencies
- Installed necessary libraries like `torch`, `librosa`, `torchaudio`, `timm`, etc.
- Utilized Kaggle environment setup for package installation.

### Data Loading
- Loaded the audio data and corresponding labels.
- Implemented efficient data loading techniques using PyTorch's `DataLoader` to handle large datasets.

### Preprocessing
- Converted audio files to spectrograms using `librosa` and `torchaudio` libraries.
- Applied normalization and other preprocessing steps to ensure the data is suitable for training.

## Data Augmentation
To enhance the model's robustness and performance, various data augmentation techniques were applied:

### Audio Augmentations
- Applied noise addition, time shifting, pitch shifting, and other audio augmentations using the `audiomentations` library.
- Implemented spectrogram augmentations like frequency masking and time masking to further diversify the training data.

### Augmentation Pipelines
- Created augmentation pipelines to apply multiple transformations sequentially to the audio data.

## Feature Engineering
Feature engineering focused on extracting meaningful features from the audio data:

### Spectrogram Features
- Extracted Mel-spectrograms and MFCC (Mel Frequency Cepstral Coefficients) from audio files.
- Utilized these features to create a rich representation of the audio data for model training.

### Statistical Features
- Calculated statistical features such as mean, variance, skewness, and kurtosis of the audio signal.

## Model Building & Training
The core of this project involves building and training a robust machine learning model:

### Transfer Learning with EfficientNet
- Leveraged the EfficientNet architecture, a state-of-the-art convolutional neural network, pre-trained on ImageNet.
- Fine-tuned the EfficientNet model to adapt it for bird sound classification by replacing the final layers to match the number of bird classes.

### Model Training
- Utilized mixed precision training with PyTorch to accelerate the training process.
- Implemented K-Fold cross-validation to ensure the model's robustness and to make the best use of available data.
- Used Adam optimizer and learning rate scheduling for optimal training performance.
- 
## ONNX Integration
To make the model more portable and optimize its performance during inference, we integrated the Open Neural Network Exchange (ONNX) format:

### Model Export to ONNX
- Converted the trained PyTorch model to ONNX format using `torch.onnx.export`.
- Ensured the model's compatibility with ONNX by handling input and output shapes properly.

### Benefits of Using ONNX
- ONNX provides interoperability across different frameworks, allowing the model to be used in various environments.
- Improved inference performance by leveraging optimized runtimes for ONNX models.

### Inference with ONNX Runtime
- Utilized ONNX Runtime for efficient model inference.
- Implemented the inference pipeline to load the ONNX model and perform predictions.
- 
## Inference
For the inference stage, the trained model was used to predict bird species from new audio recordings:

### Loading the Trained Model
- Loaded the best-performing model from the training phase.

### Prediction Pipeline
- Implemented a prediction pipeline that processes new audio data and generates predictions using the trained model.
- Applied post-processing techniques to refine the predictions and ensure accuracy

## Conclusion
This project demonstrates the application of advanced machine learning techniques to the problem of bird sound classification. By leveraging transfer learning, data augmentation, and robust feature engineering, the model achieved significant accuracy and performance. This project showcases my skills in data science, machine learning, and audio processing.

## Acknowledgments
- Kaggle for hosting the BirdCLEF 2024 competition.
- The developers of the libraries and frameworks used in this project.
#### Competition link - https://www.kaggle.com/competitions/birdclef-2024
