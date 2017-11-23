import cv2
import numpy as np
import keras
from os.path import isfile
from keras.models import Model
from keras.layers import Input
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from numpy import array
from keras.optimizers import SGD

# input image dimensions
img_rows, img_cols, channel = 128, 128, 3
k_w, k_h = 5, 5
batch_size = 128
num_classes = 5
epochs = 50
lrate = 0.0001
decay = lrate/epochs
"""
function that converts images to HSV color space and quantizes the color of the images to simplify them.

"""
def convert_to_HSV_and_quantize(img, path, K=16,
                                criteria=(cv2.TERM_CRITERIA_EPS +
                                          cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)):
    if not isfile(path):
        h = cv2.cvtColor(src=img,code=cv2.COLOR_RGB2HSV).reshape(-1,3)
        h = np.float32(h)
        ret, label, center = cv2.kmeans(data=h, K=K, bestLabels=None, criteria=criteria, attempts=10,
                                        flags=cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        qimg = res.reshape((img.shape))
        qimg = cv2.resize(qimg, (img_rows,img_cols))
        cv2.imwrite(path, qimg)
    else:
        qimg = cv2.imread(path)



    return qimg

def resize_only(img, path):
    return cv2.resize(img, (img_rows, img_cols))

def edge_detect(img, path):
    if not isfile(path):
        img = cv2.Canny(img, 100,200)
        img = cv2.resize(img, (img_rows, img_cols))

        cv2.imwrite(path, img)
    else:
        img = cv2.imread(path,0)

    img = cv2.resize(img, (img_rows,img_cols))

    return img

def Model1():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(k_w, k_h),activation='relu',input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (k_w, k_h), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
    model.add(Conv2D(128, (k_w, k_h), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (k_w, k_h), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
    model.add(Conv2D(256, (k_w, k_h), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(256, (k_w, k_h), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    return model



def Model2():
    input_shape = Input(shape=(img_rows, img_cols, 3))

    tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(input_shape)
    tower_1 = MaxPooling2D((19, 19), strides=(1, 1), padding='same')(tower_1)

    tower_2 = Conv2D(64, (9, 9), padding='same', activation='relu')(input_shape)
    tower_2 = MaxPooling2D((13, 13), strides=(1, 1), padding='same')(tower_2)

    tower_3 = Conv2D(64, (20, 20), padding='same', activation='relu')(input_shape)
    tower_3 = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(tower_3)

    merged = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)
    merged = Flatten()(merged)

    out = Dense(200, activation='relu')(merged)
    out = Dense(num_classes, activation='softmax')(out)

    model = Model(input_shape, out)
    return model

def train():

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    with open('data/train.txt') as f:
        for line in f:
            inputs = line.split()
            path = inputs[0].replace("./flower_photos", "data/flower_photos")
            label = inputs[1]
            y_train.append(label)
            image = cv2.imread(path)
            opath = inputs[0].replace("./flower_photos", "data/EDImage")
            #image = convert_to_HSV_and_quantize(image,opath)
            image = resize_only(image, opath)
            #image = edge_detect(image, opath)
            x_train.append(image)



    with open('data/val.txt')  as f:
        for line in f:
            inputs = line.split()
            path = inputs[0].replace("./flower_photos", "data/flower_photos")
            label = inputs[1]
            y_test.append(label)
            image = cv2.imread(path)
            opath = inputs[0].replace("./flower_photos", "data/EDImage")
            #image = convert_to_HSV_and_quantize(image,opath)
            image = resize_only(image, opath)
            #image = edge_detect(image, opath)
            x_test.append(image)


    from keras.preprocessing.image import ImageDataGenerator

    train_gen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        )
    test_gen = ImageDataGenerator()
    x_train = array (x_train)
    y_train = array (y_train)
    x_test  = array (x_test)
    y_test = array (y_test)



    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channel)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channel)
    input_shape = (img_rows, img_cols, channel)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    train_gen.fit(x_train)


    #model = Model1()
    #model = Model2()


    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    #base_model.save("base_model.h5")
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(256, activation='relu')(x)

    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(num_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2,
                                  patience=3, min_lr=0.00001)
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True
    model.load_weights('model3.h5')
    #sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    adm = keras.optimizers.adam(lr=lrate)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=adm,
                  metrics=['accuracy'])
    print(model.summary())

    model.fit_generator(train_gen.flow(x_train,y_train,batch_size=batch_size),
                        steps_per_epoch=len(x_train)/batch_size,epochs=epochs,
                        callbacks=[reduce_lr])




    np.random.seed(seed)
    score = model.evaluate(x_test, y_test, verbose=0)
    model.save("model3.h5")
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def test():
    x_test = []
    y_path = []
    input_shape = (img_rows, img_cols, channel)
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    #base_model.save("base_model.h5")
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(256, activation='relu')(x)

    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(5, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights('model3.h5')
    with open('data/test.txt') as f:
        for line in f:
            inputs = line.split()
            path = inputs[0].replace("./flower_photos", "data/flower_photos")
            y_path.append(path)
            image = cv2.imread(path)
            opath = path.replace("./flower_photos", "data/EDImage")
            #image = convert_to_HSV_and_quantize(image,opath)
            #image = edge_detect(image, opath)
            image = resize_only(image, opath)
            x_test.append(image)

    x_test  = array (x_test)

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channel)

    x_test = x_test.astype('float32')
    x_test /= 255

    result = model.predict(x_test, batch_size=x_test.shape[0])
    result = np.argmax(result, 1)
    import csv
    with open('project2_20411920.txt', 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(zip(y_path, result))




train()
#test()