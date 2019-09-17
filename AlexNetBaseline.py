from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

np.random.seed(333)
train_data_dir = 'data/imgs/train'
validation_data_dir = 'data/imgs/test'
nb_train_samples = 8025
nb_validation_samples = 1990
epochs = 2
batch_size = 32
img_width = 600
img_height = 450

if K.image_data_format() == 'channels_first':
	input_shape = (3, img_width, img_height)
else:
	input_shape = (img_width, img_height, 3)

model = Sequential()

#todo Add a new model here
model = Sequential([
Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'),
Conv2D(64, (3, 3), activation='relu', padding='same'),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(128, (3, 3), activation='relu', padding='same'),
Conv2D(128, (3, 3), activation='relu', padding='same',),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(256, (3, 3), activation='relu', padding='same',),
Conv2D(256, (3, 3), activation='relu', padding='same',),
Conv2D(256, (3, 3), activation='relu', padding='same',),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(512, (3, 3), activation='relu', padding='same',),
Conv2D(512, (3, 3), activation='relu', padding='same',),
Conv2D(512, (3, 3), activation='relu', padding='same',),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(512, (3, 3), activation='relu', padding='same',),
Conv2D(512, (3, 3), activation='relu', padding='same',),
Conv2D(512, (3, 3), activation='relu', padding='same',),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Flatten(),
Dense(4096, activation='relu'),
Dense(4096, activation='relu'),
Dense(7, activation='softmax')
])
# model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'))
# model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# 
# model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# 
# model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# 
# model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# 
# model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# 
# model.add(Flatten())
# model.add(Dense(4096, activation='relu'))
# model.add(Dense(4096, activation='relu'))
# 
# model.add(Dense(7, activation='softmax'))

model.compile(loss='binary_crossentropy',
			optimizer='rmsprop',
			metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1. / 255)
	#,
	# shear_range=0.2,
	# zoom_range=0.2,
	# horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
	train_data_dir,
	target_size=(img_width, img_height),
	batch_size=batch_size,
	class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
	validation_data_dir,
	target_size=(img_width, img_height),
	batch_size=batch_size,
	class_mode='categorical')

history = model.fit_generator(
	train_generator,
	steps_per_epoch=nb_train_samples // batch_size,
	epochs=epochs,
	validation_data=validation_generator,
	validation_steps=nb_validation_samples // batch_size)

#todo figure out why this doesn't work
# Y_pred = model.predict_generator(validation_generator, nb_validation_samples // batch_size+1)
# y_pred = np.argmax(Y_pred, axis=1)
# print('Confusion Matrix')
# cm = confusion_matrix(validation_generator.classes, y_pred)
# fig = plt.figure()
# plt.matshow(cm)
# plt.title('Problem 1: Confusion Matrix Digit Recognition')
# plt.colorbar()
# plt.ylabel('True Label')
# plt.xlabel('Predicated Label')
# fig.savefig('confusion_matrix_alexnet.jpg')
#fig.clf()
# print(cm)
# print('Classification Report')
# target_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
# cr = classification_report(validation_generator.classes, y_pred, target_names=target_names)
# print(cr)

model.save('VGG16.h5')

#saves figure for accuracy over time
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig('model_accuracy_alexnet.png')
plt.clf()

#saves figure for loss over time
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig('model_loss_alexnet.png')
plt.clf()

def initModel(name, input_shape):
	model = Sequential()

	if name == 'test':
		model.add(Conv2D(32, (2, 2), input_shape=input_shape))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(32, (2, 2)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(64, (2, 2)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Flatten())
		model.add(Dense(64))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		model.add(Dense(1))
		model.add(Activation('sigmoid'))
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
		model.add(Activation('sigmoid'))
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
	return model