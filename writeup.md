# **Behavioral Cloning** 

## Writeup

### Project Goals

The goals of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center_driving.jpg "Driving in the center"
[image3]: ./examples/right_side_3.jpg "Recovery image 3"
[image4]: ./examples/right_side_2.jpg "Recovery image 2"
[image5]: ./examples/right_side_1.jpg "Recovery image 1"
[image6]: ./examples/left_camera.jpg "Left side mounted camera"
[image7]: ./examples/center_camera.jpg "Center mounted camera"
[image8]: ./examples/right_camera.jpg "Right side mounted camera"


### Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing the trained convolutional neural network 
* `writeup.md` summarizing the results

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py and model.h5 files, the car can be driven autonomously around the track by starting the simulator in autonomous mode and executing 

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Parameters

#### 1. An appropriate model architecture has been employed

My model consists of a convolutional neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 (`model.py` lines 84-100). The model includes `RELU` layers to introduce nonlinearity, and the data is normalized in the model using a Keras `Lambda` layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (`model.py` line 101).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and data augmentation on images taken by three different camera perspectives (center, left, right).

For details about how I chose the training data, see the next section. 

### Documentation of Training Process

#### 1. Solution Design Approach

I used a convolutional neural network model similar to the model described in [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316). The model was used by team at NVIDIA to solve a similar problem.

To verify that my pipeline was in principle implemented correctly, I first trained the model on only 3 images: One with positive steering angle, one with negative steering angle and one with straight steering. I then let the model predict the correct steering angles for the same three images. The model was able to predict the steering angles reasonably well.

In order to gauge how well the model was working, I split the image and steering angle data from the center camera into a training and a validation set. I found that the model used by NVIDIA had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I introduced dropout layers between some of the convolutional layers.

I then ran the simulator on my model for the first time to see how well the car was driving around track 1. There were a few spots where the car left the drivable portion the track (e.g. at the dirt curve after the bridge).

I then decided to introduce the data obtained from the side cameras into the training and validation data set to specifically train the model how to behave in cases where it maneuvered itself off to the side of the road.

At the end of the process, the vehicle was able to drive autonomously around track 1 without leaving the drivable portion of the road.

#### 2. Final Model Architecture

The final model architecture (`model.py` lines 84-100) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		| Description	             					    						  | 
|-----------------------|-----------------------------------------------------------------------------| 
| Input         		| 160x320x3 BGR image   						  							  | 
| Lambda      	    	| Normalize input images to float values between -0.5 and +0.5                | 
| Cropping2D      		| Crop 70 rows of pixels from top and 20 rows of pixels from bottom of image  | 
| Convolution 1       	| Stride = 2x2, padding = valid, filter size = 5x5, filter depth = 24 		  |
| RELU					|									     									  |
| Dropout 1				| Droput probability = 0.5			     									  |
| Convolution 2       	| Stride = 2x2, padding = valid, filter size = 5x5, filter depth = 36 		  |
| RELU					|									     									  |
| Convolution 3       	| Stride = 2x2, padding = valid, filter size = 5x5, filter depth = 48 		  |
| RELU					|									     									  |
| Dropout 2				| Droput probability = 0.5			     									  |
| Convolution 4       	| Stride = 1x1, padding = valid, filter size = 3x3, filter depth = 64 		  |
| RELU					|									     									  |
| Convolution 5       	| Stride = 1x1, padding = valid, filter size = 3x3, filter depth = 64 		  |
| RELU					|									     									  |
| Dropout 3				| Droput probability = 0.5			     									  |
| Flatten   	      	|                                               							  |
| Fully Connected 1    	| Output = 100                                   							  |
| Fully Connected 2    	| Output = 50                                   							  |
| Fully Connected 3    	| Output = 10                                   							  |
| Fully Connected 4    	| Output = 1                                    							  |

#### 3. Creation of the Training Set & Training Process

At the beginning, I tried to record some data on my own by using the Udacity provided simulator. Unfortunately - I suspect partially due to the fact that I am lousy at playing computer games and partially due to latency issues - I did not suceed at driving at least one complete lap around track 1. 

I then decided to look at the sample data provided by Udacity. I used a modified version of `video.py` to create some videos of the provided set of images. In the video created from the center camera images I counted at least 8 complete laps of track 1, some clockwise and some counterclockwise, including some wandering off to the side of the road and steering back to the middle. Collecting this "drunken driver" training data was recommended in order to teach the car how to behave in cases where it maneuvered itself off to the side of the road. 

To me, the provided data seemed rich enough to train the CNN to successfully drive at least on track 1, knowing that additional data from track 2 would help the model to generalize. 

Here is an example image of center lane driving:

![alt text][image2]

Here is an example of the vehicle recovering from right side of the road back to center:

![alt text][image3]
![alt text][image4]
![alt text][image5]

In addition to the images taken by the center camera, I also used images taken by the cameras mounted on the sides of the car. In order to enable the model to successfully train on these images, the recorded steering angle has to be modified as follows: Suppose the car is driving in the center of the road. Then the image taken by the center camera will show a perfectly centered lane, while the pictures taken by the left (or right) camera will show the center of the lane to be slightly off to the right (or left) side of the image. If the recorded steering angle for a left (or right) camera picture is now adjusted by adding (or substracting) a constant, the images taken by the side cameras can be used as a second way to teach the car how to recover from wandering off the center of the road. I obtained the best results by using a correction constant of `0.2` (see `model.py`line 29 and lines 57-66).

Here is an example of the same image taken from the left, center and right perspective camera:

![alt text][image6]
![alt text][image7]
![alt text][image8]

As the sample data recorded by Udacity contains laps that are driven both clockwise as well as counter-clockwise around track 1, the risk of the model developing a bias towards steering to the left or right is very low. Therefore, image augmentation by flipping images and steering angles respectively might not have been absolutely necessary. However, it is an easy way to artificially collect more data which is why I chose to do it anyways (see `model.py` lines 52-66).

In total, the sample data recorded by Udacity contains 8,037 images. By using the side cameras and augmenting all images, I was able to obtain a total of 48,222 images that were used for training the model. 

The model preprocesses the data in the following way:
1. The data is normalized using a Keras `Lambda` layer (see `model.py` line 86).
2. As the top 70 rows of pixels of an image contain mostly landscape that is not useful to determine the correct steering angle and the bottom 25 rows of pixels mostly show the engine hood of the car, the images are cropped accordingly using a Keras `Cropping2D` layer (see `model.py` line 89).

I randomly shuffled the data set by using the Keras `train_test_split()` function and put 25% of the data into a validation set (see `model.py` line 26).

I used the training data for training the model. The validation set helped determine if the model was over- or underfitting. During training, both training loss and validation loss decreased during the first 5 epochs. I did not obtain better results for any higher number of training epochs, which is why I chose `epochs = 5` for training. Here is an exemplary console output I got during training:

```sh
502/502 [==============================] - 58s - loss: 0.0211 - val_loss: 0.0177
Epoch 2/5
502/502 [==============================] - 57s - loss: 0.0184 - val_loss: 0.0165
Epoch 3/5
502/502 [==============================] - 56s - loss: 0.0173 - val_loss: 0.0160
Epoch 4/5
502/502 [==============================] - 57s - loss: 0.0170 - val_loss: 0.0158
Epoch 5/5
502/502 [==============================] - 57s - loss: 0.0168 - val_loss: 0.0160
```

I used an adam optimizer so that manually training the learning rate was not necessary.