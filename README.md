# Sign-classifier
Python scripts for classifying traffic signs on a paper through your webcam.

The Sign-classifier is able to recognise signs if they are shown on a white paper with black borders. It is able to correctly recognise 38 signs almost perfectly.

## Installation
For this project, you need to have Python3 installed with the following packages:
* NumPy. This package was mostly used for creating and examining image matrices, but also used to get the most occurring classifier result.
* OpenCV2. To get the camera feed, we used OpenCV2. To our big surprise, this packages contained much more than just that. The most useful extra was the included perspective warp function.
* TensorFlow 2. This project used TensorFlow 2 to set up the neural network classifier.

Furthermore, this project requires a digital camera to be attached to the computer. If you experience any issues in detecting white on the paper, you should change the white threshold (line 16: delta=100) or try changing your camera settings. To select a secondary camera, you can change the camera identifier from 0 to any other (line 6: cv2.VideoCapture(0)).

Then just run main.py, and you're set.

## Further reading
The project was subdivided into the following functions subsequently:

### Fill white to find black borders
The script performs a simple fill using BFS to find the black borders. Starting at the point in the middle (the red point in the image), add neighbors of the current pixel only if they are white. The most extreme points will be saved This approach is very slow if you fill every pixel, so a few pixels will be skipped in order to reduce computation.

### Perspective warp
The image gotten after filling the white inside borders might be rotated or tilted in perspective. OpenCV has a function that can assist in turning the image, so all we needed to do was find the corners of the image.

Two cases are taken into consideration in order to simplify this task: When the downmost point is to the left of the half point of the image and when the downmost point is to the right. For the first case, the order of the points is leftmost, upmost, downmost, rightmost and for the second, upmost, rightmost, leftmost, downmost. This works really well when the image is at an angle, but if the frame is already aligned, it starts glitching. For the purpose of this project this doesn’t matter that much, as the image can just be tilted.

### Fill white to find traffic sign
A simple BFS on the white pixels is done and whenever a neighbor of one of the current pixels isn’t white, it is registered if it is an extreme point. Again, these extremities can be used to find the location of the traffic sign.

Note: pixel skipping is re-employed, since it increases performance noticeably.

### Simple crop
The location of the traffic sign is used to do a simple crop on the image. The result is a small square containing the image.

### Background noise
To prevent white background confusion, the white background gets replaced by samples taken from a gaussian distribution.

### Neural network classifier
Finally, the image is downscaled to 32x32 and fed to the neural network. The model employed 3 convolutional layers (with no pooling):
1. 32 filters and a 7x7 kernel
2. 32 filters and a 5x5 kernel
3. 64 filters and a 3x3 kernel

The last convolutional layer gets flattened and then passed to a 3 layer deep feed-forward network: the first hidden layer has a size of 128, the second 256 and the output layer has a size of 43 (the number of classes). The activation function used in all these layers is ReLU. The total accuracy reached after training is 94.5%, after training for only 2 epochs (the network saw the whole training set 2 times).

## Contributions
This project was built by [Tiberiu Iancu](https://github.com/tiberiuiancu), Sebastian-Cristian Iozu, and me [Jop Zitman](https://github.com/EndPositive).
