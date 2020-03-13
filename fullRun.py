from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential, Model, Input
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense, concatenate
from tensorflow.python.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow_model_optimization.sparsity.keras as sparsity
from models import initModel
import tensorflow as tf
import pickle as pl
#import tensorflow_hub as hub

np.random.seed(333)
train_data_dir = 'data/imgs/train'
validation_data_dir = 'data/imgs/test'
nb_train_samples = 8025
nb_validation_samples = 1990
epochs = 1 #todo change
batch_size = 32
img_width = 600
img_height = 450
begin_step = 20
end_step = np.ceil(1.0 * nb_train_samples / batch_size).astype(np.int32) * epochs
print('End step: ' + str(end_step))

pruning_params = {
      'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                   final_sparsity=0.90,
                                                   begin_step=begin_step,
                                                   end_step=end_step,
                                                   frequency=100)
}
print(K.image_data_format())
if K.image_data_format() == 'channels_first':
	input_shape = (3, img_width, img_height)
else:
	input_shape = (img_width, img_height, 3)

#names = ['alexnet', 'VGG', 'Squeezenet', 'deepsqueeze']
names = ['deepsqueeze']

for name in names:
    np.random.seed(333)
    model = initModel(name, input_shape)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(rescale=1. / 255
                                       ,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

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

    model.save('models/{0}.h5'.format(name))

    # saves figure for accuracy over time
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    plt.savefig('results/model_accuracy_{0}.png'.format(name))
    plt.clf()

    # saves figure for loss over time
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    plt.savefig('results/model_loss_{0}.png'.format(name))
    plt.clf()

    if name == 'deepsqueeze':
        pruned = sparsity.prune_low_magnitude(model, **pruning_params)
        np.random.seed(333)
        pruned.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        results = pruned.evaluate_generator(validation_generator)
        results = [str(i) for i in results]

        model.save('models/{0}_pruned.h5'.format(name))

        resultsFile = open("results/{0}_pruned.txt".format(name), 'w')
        resultsFile.writelines(results)
        resultsFile.close()

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                               tf.lite.OpsSet.SELECT_TF_OPS]

        tflite_model = converter.convert()

        open("models/{0}_compressed.tflite".format(name), "wb").write(tflite_model)
        quantized = tf.lite.TFLiteConverter.from_saved_model("models/{0}_compressed.tflite".format(name))
        np.random.seed(333)
        quantized.compile(loss='binary_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy'])
        results = quantized.evaluate_generator(validation_generator)
        results = [str(i) for i in results]
        resultsFile = open("results/{0}_compressed.txt".format(name), 'w')
        resultsFile.writelines(results)
        resultsFile.close()


