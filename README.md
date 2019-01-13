# Convolutional-Neural-Network
Implementation of CNN in Matlab
Scripts

CNN.m: Implementation of the convolutional neural network architecture which consists of 2 convolutional layer, 2 pooling layers, 1 fully-connected layer 
load_features.m: Preparing training and validation sets for the features
load_labels.m: Preparing training and validation sets for the labels
visualization.m: Script to visualize the effects of each of the layers on the images.

To use the visualization.m script:
1) run the CNN script by typing 'CNN' on the MATLAB command prompt
2) run the visualization script using the following command 'visualization(CNN, 'layer_name')

layer_name =  {conv1, conv2, maxpool1, maxpool2, fullconnect1}
