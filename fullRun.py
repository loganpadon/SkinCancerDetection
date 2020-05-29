from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential, Model, Input
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense, concatenate
from tensorflow.python.keras import backend as K
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.python.keras.losses import CategoricalCrossentropy
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_model_optimization.sparsity.keras as sparsity
from models import initModel
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import pickle as pl
import math
from sklearn.utils import compute_class_weight
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

np.random.seed(333)
train_data_dir = 'data/imgs/train'
validation_data_dir = 'data/imgs/test'
nb_train_samples = 3979
nb_validation_samples = 985
epochs = 100 #todo change
batch_size = 1
img_width = 600
img_height = 450
begin_step = 20
end_step = np.ceil(1.0 * nb_train_samples / batch_size).astype(np.int32) * epochs
classes = ['akiec','bcc','bkl','df','mel','nv','vasc']
class_count = len(classes)
print('End step: ' + str(end_step))
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)

def create_class_weight(labels_dict):
    # weights = compute_class_weight(labels_dict, np.array(range(0,7)), labels_dict)
    # keys = labels_dict.keys()
    class_weight = {clsID : 5367/numIMG for clsID, numIMG in labels_dict}
    return class_weight


def plot_classification_report(cr, title='Classification Report ', with_avg_total=False, cmap=plt.cm.Blues):

    lines = cr.split('\n')

    classes = []
    plotMat = []
    for line in lines[2 : (len(lines) - 3)]:
        #print(line)
        t = line.split()
        # print(t)
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        print(v)
        plotMat.append(v)

    if with_avg_total:
        aveTotal = lines[len(lines) - 1].split()
        classes.append('avg/total')
        vAveTotal = [float(x) for x in t[1:len(aveTotal) - 1]]
        plotMat.append(vAveTotal)


    plt.imshow(plotMat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(3)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('Classes')
    plt.xlabel('Measures')


#class_weight = {0: 271, 1: 412, 2: 869, 3: 88, 4: 899, 5: 1321, 6: 119}
#class_weight = [271, 412, 869, 88, 899, 5367, 119]
class_weight = {0: 2.69, 1: 2.26, 2: 1.52, 3: 3.81, 4: 1.49, 5: 1.1, 6: 3.51}
#class_bias = np.array([-3.353, -2.917, -2.108, -4.502, -2.07, 0.703, -4.196])
# class_weight = create_class_weight(class_weight)

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
#names = ['Squeezenet', 'deepsqueeze', 'son_of_deepsqueeze', 'the_big_squeeze','alexnet', 'inceptionV3']
names = ['alexnet']

for name in names:
    np.random.seed(333)
    model = initModel(name, input_shape)
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy'])
    model.compile(
        optimizer=sgd,
        loss=CategoricalCrossentropy(),
        metrics=[categorical_accuracy, top_2_accuracy, top_3_accuracy])

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
        validation_steps=nb_validation_samples // batch_size, class_weight=class_weight)

    model.save('models/{0}.h5'.format(name))

    # saves figure for accuracy over time
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    plt.savefig('results/{0}/model_accuracy_{0}.png'.format(name))
    plt.clf()

    # saves figure for loss over time
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    plt.savefig('results/{0}/model_loss_{0}.png'.format(name))
    plt.clf()

    # Confution Matrix and Classification Report
    Y_pred = model.predict_generator(validation_generator, nb_validation_samples // batch_size + 1)
    y_pred = np.argmax(Y_pred, axis=1)
    con_mat = tf.math.confusion_matrix(validation_generator.classes, y_pred)
    con_mat_norm = np.around(con_mat / con_mat.sum(axis=1)[:, np.newaxis], decimals=2) #con_mat.astype('float')
    con_mat_df = pd.DataFrame(con_mat_norm,
                              index=classes,
                              columns=classes)
    figure = plt.figure(figsize=(class_count, class_count))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.show()
    plt.savefig('results/{0}/confusion_matrix_{0}.png'.format(name))

    clf_report = classification_report(validation_generator.classes, y_pred, labels=np.arange(7), target_names=classes,
                                       output_dict=True)
    heatmap = sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
    fig = heatmap.get_figure()
    fig.savefig('results/{0}/classification_report_{0}.png'.format(name))

    #if name == 'the_big_squeeze':
    if True:
        pruned = sparsity.prune_low_magnitude(model, **pruning_params)
        np.random.seed(333)
        pruned.compile(loss='categorical_crossentropy',
                      optimizer='sgd',
                      metrics=['accuracy'])
        results = pruned.evaluate_generator(validation_generator)
        results = [str(i) for i in results]

        pruned.save('models/{0}_pruned.h5'.format(name))

        resultsFile = open("results/{0}/{0}_pruned.txt".format(name), 'w')
        resultsFile.writelines(results)
        resultsFile.close()

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                               tf.lite.OpsSet.SELECT_TF_OPS]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()

        open("models/{0}_compressed.tflite".format(name), "wb").write(tflite_model)
        # quantized = tf.lite.TFLiteConverter.from_saved_model("models/{0}_compressed.tflite".format(name))
        # np.random.seed(333)
        # quantized.compile(loss='binary_crossentropy',
        #                optimizer='adam',
        #                metrics=['accuracy'])
        # results = quantized.evaluate_generator(validation_generator)
        # results = [str(i) for i in results]
        # resultsFile = open("results/{0}_compressed.txt".format(name), 'w')
        # resultsFile.writelines(results)
        # resultsFile.close()


