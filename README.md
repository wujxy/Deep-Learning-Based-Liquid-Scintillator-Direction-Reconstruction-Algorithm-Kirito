# Deep-Learning-Based-Liquid-Scintillator-Direction-Reconstruction-Algorithm-Kirito
The goal of this project is to use visual information from a liquid scintillator detector to identify Cherenkov rings of charged particles and reconstruct the particle directions with CNN framework.

A spherical liquid scintillator detector is similar to a camera, where each PMT on the periphery corresponds to a pixel point, and the entire spherical surface acts as a two-dimensional image. This program converts the hit information from the spherical liquid scintillator detector into an n×n two-dimensional image through Mercator projection, which is then used as the original input.
The Cherenkov ring recognition model Kirito, based on Convolutional Neural Networks (CNN), consists of 5 convolutional layers, one flattening layer, and one fully connected layer. The input is a two-dimensional pixel image, and the output is the predicted coordinates of the Cherenkov ring’s center pixel.
The project have three files, the first one is self-defined dataset,the second one is definition of Kirito model . the final one is to train moel.

If you want run full process, you should transform your data to pixel by Dataset.py, and input dataset into model to train model by Kirito_train.py.
