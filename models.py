from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential, Model, Input
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense, concatenate
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_model_optimization.sparsity.keras as sparsity
import numpy as np
from tensorflow.keras.initializers import Constant

class_bias = Constant(np.array([-3.353, -2.917, -2.108, -4.502, -2.07, 0.703, -4.196]))

def fire_module(x, fire_id, squeeze=16, expand=64):
	sq1x1 = "squeeze1x1"
	exp1x1 = "expand1x1"
	exp3x3 = "expand3x3"
	relu = "relu_"
	s_id = 'fire' + str(fire_id) + '/'

	if K.image_data_format() == 'channels_first':
		channel_axis = 1
	else:
		channel_axis = 3

	x = Conv2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
	x = Activation('relu', name=s_id + relu + sq1x1)(x)

	left = Conv2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
	left = Activation('relu', name=s_id + relu + exp1x1)(left)

	right = Conv2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
	right = Activation('relu', name=s_id + relu + exp3x3)(right)

	x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
	return x

def initModel(name, input_shape):
	model = Sequential()

	if name == 'test':
		model.add(Dense(512, input_shape=input_shape))
		model.add(Activation("relu"))
		model.add(Dense(7))
		model.add(Activation("sigmoid"))
	elif name == 'alexnet':
		# 1st Convolutional Layer
		model.add(Conv2D(filters=96, input_shape=input_shape, kernel_size=(11, 11), strides=(4, 4), padding='valid'))
		model.add(Activation('relu'))
		# Max Pooling
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

		# 2nd Convolutional Layer
		model.add(Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding='valid'))
		model.add(Activation('relu'))
		# Max Pooling
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

		# 3rd Convolutional Layer
		model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
		model.add(Activation('relu'))

		# 4th Convolutional Layer
		model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
		model.add(Activation('relu'))

		# 5th Convolutional Layer
		model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
		model.add(Activation('relu'))
		# Max Pooling
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

		# Passing it to a Fully Connected layer
		model.add(Flatten())
		# 1st Fully Connected Layer
		model.add(Dense(4096, input_shape=input_shape))
		model.add(Activation('relu'))
		# Add Dropout to prevent overfitting
		model.add(Dropout(0.4))

		# 2nd Fully Connected Layer
		model.add(Dense(4096))
		model.add(Activation('relu'))
		# Add Dropout
		model.add(Dropout(0.4))

		# 3rd Fully Connected Layer
		model.add(Dense(1000))
		model.add(Activation('relu'))
		# Add Dropout
		model.add(Dropout(0.4))

		# Output Layer
		model.add(Dense(7))
		model.add(Activation('softmax'))
	elif name == 'VGG':
		model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'))
		model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
		model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
		model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
		model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
		model.add(Conv2D(512, (3, 3), activation='relu', padding ='same'))
		model.add(Conv2D(512, (3, 3), activation='relu', padding ='same'))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		model.add(Conv2D(512, (3, 3), activation='relu', padding ='same'))
		model.add(Conv2D(512, (3, 3), activation='relu', padding ='same'))
		model.add(Conv2D(512, (3, 3), activation='relu', padding ='same'))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		model.add(Flatten())
		model.add(Dense(4096, input_shape=input_shape, activation='relu'))
		model.add(Dense(4096, activation='relu'))

		model.add(Dense(7, activation='softmax'))

	elif name == 'Squeezenet':
		img_input = Input(shape=input_shape)

		x = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(img_input)
		x = Activation('relu', name='relu_conv1')(x)
		x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

		x = fire_module(x, fire_id=2, squeeze=16, expand=64)
		x = fire_module(x, fire_id=3, squeeze=16, expand=64)
		x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

		x = fire_module(x, fire_id=4, squeeze=32, expand=128)
		x = fire_module(x, fire_id=5, squeeze=32, expand=128)
		x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

		x = fire_module(x, fire_id=6, squeeze=48, expand=192)
		x = fire_module(x, fire_id=7, squeeze=48, expand=192)
		x = fire_module(x, fire_id=8, squeeze=64, expand=256)
		x = fire_module(x, fire_id=9, squeeze=64, expand=256)
		x = Dropout(0.5, name='drop9')(x)

		x = Conv2D(7, (1, 1), padding='valid', name='conv10')(x) #uses classes bias_initializer=class_bias
		x = Activation('relu', name='relu_conv10')(x)
		x = GlobalAveragePooling2D()(x)
		x = Activation('softmax', name='loss')(x)

		# Ensure that the model takes into account
		# any potential predecessors of `input_tensor`.
		inputs = img_input

		model = Model(inputs, x, name='squeezenet')

	elif name == 'deepsqueeze':
		img_input = Input(shape=input_shape)

		x = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(img_input)
		x = Activation('relu', name='relu_conv1')(x)
		x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

		x = fire_module(x, fire_id=2, squeeze=16, expand=64)
		x = fire_module(x, fire_id=3, squeeze=16, expand=64)
		x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

		x = fire_module(x, fire_id=4, squeeze=32, expand=128)
		x = fire_module(x, fire_id=5, squeeze=32, expand=128)
		x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

		x = fire_module(x, fire_id=6, squeeze=32, expand=128)
		x = fire_module(x, fire_id=7, squeeze=32, expand=128)
		x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool7')(x)

		x = fire_module(x, fire_id=8, squeeze=32, expand=128)
		x = fire_module(x, fire_id=9, squeeze=32, expand=128)
		x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool9')(x)

		x = fire_module(x, fire_id=14, squeeze=32, expand=128)
		x = fire_module(x, fire_id=15, squeeze=32, expand=128)
		x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool11')(x)

		x = fire_module(x, fire_id=10, squeeze=48, expand=192)
		x = fire_module(x, fire_id=11, squeeze=48, expand=192)
		x = fire_module(x, fire_id=12, squeeze=64, expand=256)
		x = fire_module(x, fire_id=13, squeeze=64, expand=256)
		x = Dropout(0.5, name='drop13')(x)

		x = Conv2D(7, (1, 1), padding='valid', name='conv10')(x)  # uses classes
		x = Activation('relu', name='relu_conv10')(x)
		# x = GlobalAveragePooling2D()(x)
		x = GlobalAveragePooling2D(data_format='channels_last')(x)
		x = Activation('softmax', name='loss')(x)

		# Ensure that the model takes into account
		# any potential predecessors of `input_tensor`.
		inputs = img_input

		model = Model(inputs, x, name='deepsqueeze')
	elif name == 'son_of_deepsqueeze':
		img_input = Input(shape=input_shape)

		x = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(img_input)
		x = Activation('relu', name='relu_conv1')(x)
		x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

		x = fire_module(x, fire_id=2, squeeze=16, expand=64)
		x = fire_module(x, fire_id=3, squeeze=16, expand=64)
		x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

		x = fire_module(x, fire_id=4, squeeze=32, expand=128)
		x = fire_module(x, fire_id=5, squeeze=32, expand=128)
		x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

		x = fire_module(x, fire_id=6, squeeze=48, expand=192)
		x = fire_module(x, fire_id=7, squeeze=48, expand=192)
		x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool7')(x)

		x = fire_module(x, fire_id=8, squeeze=64, expand=256)
		x = fire_module(x, fire_id=9, squeeze=64, expand=256)
		x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool9')(x)

		x = fire_module(x, fire_id=14, squeeze=80, expand=320)
		x = fire_module(x, fire_id=15, squeeze=80, expand=320)
		x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool11')(x)

		x = fire_module(x, fire_id=10, squeeze=96, expand=384)
		x = fire_module(x, fire_id=11, squeeze=96, expand=384)
		x = fire_module(x, fire_id=12, squeeze=112, expand=448)
		x = fire_module(x, fire_id=13, squeeze=112, expand=448)
		x = Dropout(0.5, name='drop13')(x)

		x = Conv2D(7, (1, 1), padding='valid', name='conv10')(x)  # uses classes
		x = Activation('relu', name='relu_conv10')(x)
		# x = GlobalAveragePooling2D()(x)
		x = GlobalAveragePooling2D(data_format='channels_last')(x)
		x = Activation('softmax', name='loss')(x)

		# Ensure that the model takes into account
		# any potential predecessors of `input_tensor`.
		inputs = img_input

		model = Model(inputs, x, name='son_of_deepsqueeze')

	elif name == 'the_big_squeeze': #snow_squeeze
		img_input = Input(shape=input_shape)

		x = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(img_input)
		x = Activation('relu', name='relu_conv1')(x)
		x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

		x = fire_module(x, fire_id=2, squeeze=16, expand=64)
		x = fire_module(x, fire_id=3, squeeze=16, expand=64)
		x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

		x = fire_module(x, fire_id=4, squeeze=32, expand=128)
		x = fire_module(x, fire_id=5, squeeze=32, expand=128)
		x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

		x = fire_module(x, fire_id=6, squeeze=48, expand=192)
		x = fire_module(x, fire_id=7, squeeze=48, expand=192)
		x = fire_module(x, fire_id=8, squeeze=64, expand=256)
		x = fire_module(x, fire_id=9, squeeze=64, expand=256)
		x = Dropout(0.5, name='drop9')(x)

		x = Flatten()(x)
		x = Dense(7, activation='softmax')(x)


		# Ensure that the model takes into account
		# any potential predecessors of `input_tensor`.
		inputs = img_input

		model = Model(inputs, x, name='the_big_squeeze')
	elif name == 'inceptionV3':
		InceptionV3(include_top=False, input_shape=input_shape, classes=7)
	return model