# A tensorflow template for deep learning projects

This is a sample deep learning model template for most of the deep learning projects.
The template is heavily inspired by the official documentation from tensorflow https://www.tensorflow.org/tutorials/deep_cnn

## Code organization
data_input.py - Routine for read the data, decode it and return as TFRecord

model.py - The core model, including Model inputs `inputs()`, Model prediction `inference()`, Model training `loss()` and `train()`

1. Model inputs:
    `inputs()` adds operations that read and preprocess the data for training and evaluation
2. Model prediction:
    `inference()` adds operations that perform inference, i.e. classification
3. Model training:
    `loss()` and `train()` add operations that compute the loss, gradients, variable updates and visualization summaries

