import csv
import os
from scipy import ndimage
import numpy as np
import cv2
import sklearn

log_path = "/opt/carnd_p3/data/driving_log.csv"
image_path = "/opt/carnd_p3/data/IMG/"

lines = []
n = 0

# Read in csv file
with open(log_path) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        n += 1
        if n == 1:
            pass # Do not read in the first line
        else:
            lines.append(line) 

# Split data into training and validation set
from sklearn.model_selection import train_test_split
train_lines, valid_lines = train_test_split(lines, test_size=0.25)

batch_size = 12
correction = 0.2

# Generator returning batches of shuffled X and y
def generator(lines, batch_size):
    n_lines = len(lines)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(lines)
        for offset in range(0, n_lines, batch_size):
            batch_lines = lines[offset:offset+batch_size]

            images = []
            measurements = []
            
            for batch_line in batch_lines:
                source_paths = batch_line[0:3] # Load image taken by center, left and right camera
                i = 0
                for source_path in source_paths:
                    filename = source_path.split('/')[-1]
                    current_path = image_path + filename
                    image = ndimage.imread(current_path)
                    # images.append(image)
                    measurement = float(batch_line[3]) # Get corresponding measurement from driving log
                    
                    if i == 0:
                        images.append(image)
                        images.append(np.fliplr(image))
                        measurements.append(measurement)
                        measurements.append(-measurement)
                    elif i == 1: # left camera
                        images.append(image)
                        images.append(np.fliplr(image))
                        measurements.append(measurement+correction)
                        measurements.append(-(measurement+correction))
                    elif i == 2: # right camera
                        images.append(image)
                        images.append(np.fliplr(image))
                        measurements.append(measurement-correction)
                        measurements.append(-(measurement-correction))
                    i += 1
                    
                    # measurements.append(measurement)

            X = np.array(images)
            y = np.array(measurements)
            yield sklearn.utils.shuffle(X, y)
            
train_generator = generator(train_lines, batch_size=batch_size)
valid_generator = generator(valid_lines, batch_size=batch_size)
        
from keras.models import Sequential, load_model
from keras.layers import Cropping2D, Lambda, Conv2D, Flatten, Dense, Dropout

# Start with pre-trained model
train_from_scratch = True

if train_from_scratch:
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3))) # Normalization Layer
    model.add(Cropping2D(cropping=((70,25),(0,0)))) # Crop 50 pixel rows from top and 20 from bottom
    model.add(Conv2D(24,(5,5),strides=(2,2),activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(36,(5,5),strides=(2,2),activation='relu'))
    model.add(Conv2D(48,(5,5),strides=(2,2),activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
else:
    model = load_model('model.h5')

steps_per_epoch_train = len(train_lines)/batch_size
steps_per_epoch_valid = len(valid_lines)/batch_size

history = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch_train, epochs=5, verbose=1, validation_data=valid_generator, validation_steps=steps_per_epoch_valid)

model.save('model.h5')