# Kaggle Toxic Comment classification

This is a deep learning model for the Kaggle Toxic Comment classification. 

The key of this project is not only creating a model for classifying comments, but also creating a project template following the best practice of tensorflow. 

The template is heavily inspired by the official documentation from tensorflow https://www.tensorflow.org/tutorials/deep_cnn

## Code organization
data_input.py - Routine for read the data, decode it and return as TFRecord

model.py - The core model, including Model prediction `inference()`, Model training `loss()` and `train()`

1. Model prediction:
    `inference()` adds operations that perform inference, i.e. classification
2. Model training:
    `loss()` and `train()` add operations that compute the loss, gradients, variable updates and visualization summaries

