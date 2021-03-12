import numpy as np

from keras import layers
from keras.layers import Input, Dense, Activation, Flatten, Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers import ZeroPadding2D, AveragePooling2D, GlobalAveragePooling2D, BatchNormalization

from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.engine.topology import get_source_inputs


input = Input(shape=(224,224,3))


def identity_block(input, filters, stage, block):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    X_shortcut = input

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size= (3, 3), padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, kernel_size=(1,1), strides= (1,1), padding = 'valid', name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    x = layers.add([x, X_shortcut])
    x = Activation('relu')(x)
    return x



def conv_block(input, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    short_cut = input

    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size=(3, 3), padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)


    ### shortcut ###
    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(short_cut)
    shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(input_shape=(224,224,3), classes=1000):
    # Determine proper input shape
    X_input = Input(input_shape)

    x = ZeroPadding2D((3, 3))(X_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, [64, 64, 256], stage=2, block='b')
    x = identity_block(x,  [64, 64, 256], stage=2, block='c')

    x = conv_block(x, [128, 128, 512], stage=3, block='a')
    x = identity_block(x,  [128, 128, 512], stage=3, block='b')
    x = identity_block(x,  [128, 128, 512], stage=3, block='c')
    x = identity_block(x,  [128, 128, 512], stage=3, block='d')

    x = conv_block(x,  [256, 256, 1024], stage=4, block='a')
    x = identity_block(x,  [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x,  [256, 256, 1024], stage=4, block='e')
    x = identity_block(x,  [256, 256, 1024], stage=4, block='f')

    x = conv_block(x,  [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x,  [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc1000')(x)

    # Create model.
    model = Model(inputs = X_input, outputs = x, name='resnet50')

    return model


if __name__ == '__main__':
    model = ResNet50()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()