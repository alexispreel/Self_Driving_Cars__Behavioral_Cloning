import csv
import cv2
import os
import zipfile
import numpy as np

# Unzipping files from training set 1
try:
    if not os.path.exists('driving_log_1.csv'):
        zip_ref = zipfile.ZipFile('./training_data_1.zip', 'r')
        zip_ref.extractall()
        print('Files unzipped')
    else:
        print('Files already unzipped')

    if os.path.exists('driving_log.csv'):
        os.rename('driving_log.csv', 'driving_log_1.csv')
        os.rename('IMG', 'IMG_1')
        print('Files renamed')
    else:
        print('Files already renamed')
    
except:
    print('No file named training_data_1.csv')


# Unzipping files from training set 2
try:
    if not os.path.exists('driving_log_2.csv'):
        zip_ref = zipfile.ZipFile('./training_data_2.zip', 'r')
        zip_ref.extractall()
        print('Files unzipped')
    else:
        print('Files already unzipped')

    if os.path.exists('driving_log.csv'):
        os.rename('driving_log.csv', 'driving_log_2.csv')
        os.rename('IMG', 'IMG_2')
        print('Files renamed')
    else:
        print('Files already renamed')
        
except:
    print('No file named training_data_2.csv')


# Unzipping files from training set misc
try:
    if not os.path.exists('driving_log_misc.csv'):
        zip_ref = zipfile.ZipFile('./training_data_misc.zip', 'r')
        zip_ref.extractall()
        print('Files unzipped')
    else:
        print('Files already unzipped')
    
    if os.path.exists('driving_log.csv'):
        os.rename('driving_log.csv', 'driving_log_misc.csv')
        os.rename('IMG', 'IMG_misc')
        print('Files renamed')
    else:
        print('Files already renamed')
    
except:
    print('No file named training_data_misc.csv')



# Variables to crop images (defined for 160x320 images)
crop = True
crop_v = (45,22)
crop_h = (0,0)

# Variables to resize images
resize = True
target_size = (200,66)

# Variables to update image brightness
brightness = False
a = np.array([1.25])
b = np.array([-100.0])

def feed_training_data(lines, folder, steering_correction=[0.0,0.25,-0.25]):
    # Initializing array of images
    images = []

    # Initializing array of steering values
    steerings = []
    
    # Browsing driving log
    for line in lines: 
        # Reading steering measurement
        steering = float(line[3])

        # For center, left and right camera
        for i in range(3):
            # Reading path
            source_path = line[i]

            # Getting file name from path
            filename = source_path.split('\\')[-1]
            current_path = folder + '/' + filename

            # Loading raw image
            image = cv2.imread(current_path)
            
            # Turning back from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Cropping image
            if crop:
                image = image[crop_v[1] : image.shape[0] - crop_v[0], crop_h[1] : image.shape[1] - crop_h[0]]

            # Resizing image
            if resize:
                image = cv2.resize(image, target_size)

            # Updating brightness
            if brightness:
                cv2.add(image, b, image)
                cv2.multiply(image, a, image)

            # Appending image to set
            images.append(image)

            # Appending steering measurement to set
            steerings.append(steering + steering_correction[i])

    return np.array(images), np.array(steerings)
	
	
	
# Building training set - Data collected on first track
try:
    lines = []
    with open('driving_log_1.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    X_train, y_train = feed_training_data(lines, 'IMG_1')

except:
    print('No file named driving_log_1.csv')


# Building training set - Data collected on second track
try:
    lines = []
    with open('driving_log_2.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    X_train_temp, y_train_temp = feed_training_data(lines, 'IMG_2')
    X_train = np.concatenate((X_train, X_train_temp), axis=0)
    y_train = np.concatenate((y_train, y_train_temp), axis=0)

except:
    print('No file named driving_log_2.csv')


# Building training set - Adding recovery data from shoulders
try:
    lines = []
    with open('driving_log_misc.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
            
    X_train_temp, y_train_temp = feed_training_data(lines, 'IMG_misc')
    X_train = np.concatenate((X_train, X_train_temp), axis=0)
    y_train = np.concatenate((y_train, y_train_temp), axis=0)
    
except:
    print('No file named driving_log_misc.csv')


# Printing statistics
n_train = len(X_train)
print('Number of images: ', n_train)
img_shape = X_train[0].shape
print('Shape of images: ', img_shape)
print('image ratio = ', img_shape[0]/img_shape[1])



# Filtering training set according to steering measurement
indices = np.where(abs(y_train) >= 0.0)
X_train = X_train[indices]
y_train = y_train[indices]

# Printing statistics
n_train = len(X_train)
print('Number of images: ', n_train)
img_shape = X_train[0].shape
print('Shape of images: ', img_shape)
print('image ratio = ', img_shape[0]/img_shape[1])



# To visualize sign image given index
def visualize_sign_index(feature_set, label_set, index, cmap=None):
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    %matplotlib inline
    
    image = feature_set[index]
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Printing image index, sign image, sign label and name
    print('Index picked:', index)
    print('Label:', label_set[index])
    plt.figure()
    plt.imshow(image, cmap=cmap)



# Augmenting training set - Horizontal flip

# Selecting images for horizontal flip
indices = np.where(abs(y_train) >= 0.5)
p = 1.0
if p == 1.0:
    print('Entire set selected for augmentation')
    X_train_1 = X_train[indices]
    y_train_1 = y_train[indices]
else:
    import random
    indices = np.random.choice(len(indices), int(np.ceil(p * len(indices))))
    X_train_1 = X_train[indices]
    y_train_1 = y_train[indices]
print('Number of images selected for horizontal flip: ', len(X_train_1))
    
# Visualizing image before augmentation
import random
random_index = random.randint(0, len(X_train_1))
visualize_sign_index(X_train_1, y_train_1, random_index)



# Horizontal flip
def flip_h(image_data, label_data):
    import random
    import numpy as np
    # Initializing output
    im_out = []
    lb_out = []
    for k in range(len(image_data)):
        # Flipping image horizontally
        im = np.fliplr(image_data[k])
        # Appending image ouput
        im_out.append(im)
        # Appending label ouput
        lb_out.append(-label_data[k])
    return im_out, lb_out



# Flipping images horizontally
X_train_1, y_train_1 = flip_h(X_train_1, y_train_1)

# Visualizing same image after augmentation
visualize_sign_index(X_train_1, y_train_1, random_index)



# Augmenting training set - Horizontal translation

# Selecting p% of images for horizontal translation
p = 0.2
if p == 1.0:
    print('Entire set selected for augmentation')
    X_train_2 = X_train
    y_train_2 = y_train
else:
    import random
    indices = np.random.choice(n_train, int(np.ceil(p * n_train)))
    X_train_2 = X_train[indices]
    y_train_2 = y_train[indices]
print('Number of images selected for translation: ', len(X_train_2))

# Visualizing image before augmentation
import random
random_index = random.randint(0, len(X_train_2))
visualize_sign_index(X_train_2, y_train_2, random_index)



# Horizontal translation
def trans_h(image_data, label_data):
    import random
    import numpy as np
    import cv2
    
    # Initializing output
    im_out = []
    lb_out = []
    for k in range(len(image_data)):
        
        # Getting image and steering
        im = np.float32(image_data[k])
        st = label_data[k]

        # Generating random translation factors
        trans_max = im.shape[1] / 4
        trans_x = trans_max * (np.random.uniform() - 0.5)
        trans_y = 0

        # Adjusting steering
        st = st + trans_x / trans_max * 2 * 0.2

        # Running translation
        M = np.float32([[1, 0, trans_x],[0, 1, trans_y]])
        im = cv2.warpAffine(im, M, (im.shape[1], im.shape[0]))
        
        im = np.int8(im)
        
        # Appending image ouput
        im_out.append(im)
        # Appending label ouput
        lb_out.append(st)
        
    
    return im_out, lb_out



# Translating
X_train_2, y_train_2 = trans_h(X_train_2, y_train_2)

# Visualizing same image after augmentation
visualize_sign_index(X_train_2, y_train_2, random_index)



# Adding to training set
try:
    X_train = np.concatenate((X_train, X_train_1), axis=0)
    y_train = np.concatenate((y_train, y_train_1), axis=0)
except:
    print('No X_train_1')
    
try:
    X_train = np.concatenate((X_train, X_train_2), axis=0)
    y_train = np.concatenate((y_train, y_train_2), axis=0)
except:
    print('No X_train_2')
    
img_shape = X_train[0].shape

# Printing statistics
print('number of images = ', len(X_train))
print('image shape = ', img_shape)
print('image ratio = ', img_shape[0]/img_shape[1])



# NVidia Model
import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Cropping2D
from keras.layers import Lambda, Input, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.regularizers import l2


# Initializing model
model = Sequential()


# Normalizing image data, centering around zero with small standard deviation
model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=img_shape, output_shape=img_shape))


# Convolutional layers: kernel=5x5, stride=2x2
model.add(Convolution2D(24,5,5, subsample=(2,2), border_mode='valid', W_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(36,5,5, subsample=(2,2), border_mode='valid', W_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(48,5,5, subsample=(2,2), border_mode='valid', W_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# Convolutional layers: kernel=3x3, stride=1x1
model.add(Convolution2D(64,3,3, subsample=(1,1), border_mode='valid', W_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3, subsample=(1,1), border_mode='valid', W_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# Flattening images
model.add(Flatten())

# Fully connected layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# Fully connected layer
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# Fully connected layer
model.add(Dense(10))
model.add(Activation('relu'))

# Fully connected layer
model.add(Dense(1))


# Defining loss function and optimizer
model.compile(loss='mse', optimizer='adam')



from sklearn.model_selection import train_test_split

# Isolating part of training set for future validation (20% here)
X_train, X_valid,  y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

print('Number of images for training: ', len(X_train))
print('Number of images for validation: ', len(X_valid))



def generator(X_data, y_data, batch_size=128):
    import random
    while True: # Loop forever so the generator never terminates
        
        indices = np.random.choice(X_data.shape[0], batch_size, replace=False)
        X_batch = X_data[indices, :, :, :]
        y_batch = y_data[indices]
        
        yield X_batch, y_batch



# compile and train the model using the generator function
train_gen = generator(X_train, y_train, batch_size=128)
valid_gen = generator(X_valid, y_valid, batch_size=128)



# Creating checkpoint for future iterations
checkpoint = keras.callbacks.ModelCheckpoint('checkpoints/weights2.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', save_weights_only=False)

# Training data
samples_per_epoch = len(X_train) - len(X_train) % 128 + 128
history_object = model.fit_generator(train_gen, samples_per_epoch=samples_per_epoch, nb_epoch=50, verbose=2, validation_data=valid_gen, nb_val_samples=len(X_valid), max_q_size=25, pickle_safe=True, nb_worker=4, callbacks=[checkpoint])



# Saving model to file
model_name = 'model.h5'
model.save(model_name)

print('Model file size: ', os.path.getsize(model_name))



# Plotting training and validation loss for each epoch
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Model mean squared error loss')
plt.ylabel('Mean squared error loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper right')
plt.show()
