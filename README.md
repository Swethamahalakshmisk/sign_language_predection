Sign Language MNIST Classification

This project demonstrates a Convolutional Neural Network (CNN) to classify images of American Sign Language letters (A-Z) using the Sign Language MNIST dataset. The model can identify 26 unique classes, representing each letter of the alphabet. The CNN architecture has multiple convolutional layers followed by max pooling, dropout for regularization, and dense layers for classification.

Project Structure

sign_mnist_train.csv: Training dataset of hand gestures representing letters A-Z.
sign_mnist_test.csv: Testing dataset of hand gestures representing letters A-Z.

Getting Started
Prerequisites:

pandas
numpy
tensorflow
scikit-learn
matplotlib

The CNN model consists of the following layers:

Convolutional Layer: Extracts features from images.
MaxPooling: Reduces spatial dimensions and controls overfitting.
Flatten: Converts the 3D feature maps into 1D for dense layers.
Dense Layer: Performs classification with softmax activation for 26 output classes.
Training and Evaluation
The model is trained over 5 epochs using categorical cross-entropy as the loss function and Adam optimizer. After training, the model’s accuracy is evaluated on the test set.

Visualizations

Training and Validation Accuracy: Plot to show accuracy improvements over epochs.
Training and Validation Loss: Plot to show loss reduction over epochs.

Results

The model typically achieves an accuracy of ~0.90% on the test set , indicating the model’s ability to classify sign language gestures.
