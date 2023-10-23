from keras.layers import add
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D, Activation, average
from keras.layers import Add, Subtract, Multiply, Average, Maximum, Minimum, Concatenate,Convolution2D
from keras.layers import merge, BatchNormalization, SpatialDropout2D, Dropout,Flatten, Dense, Reshape, GlobalAveragePooling2D
from keras.models import Model
def conv3x3(x, out_filters, strides=(1, 1), name=''):
    x = Conv2D(out_filters,3, padding='same', strides=strides, use_bias=False, kernel_initializer='he_normal',
               name=name)(x)
    return x

def basic_Block(input, out_filters, strides=(1, 1), with_conv_shortcut=False, name=''):
    x = conv3x3(input, out_filters, strides=strides)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = conv3x3(x, out_filters)
    x = BatchNormalization(axis=3)(x)
    if with_conv_shortcut:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal',
                         )(input)
        residual = BatchNormalization(axis=3)(residual)
        x = add([x, residual])
    else:
        x = add([x, input])
    x = Activation('relu')(x)
    return x

def Conv2D_d(input, outdim, dilated=1, is_batchnorm=True, name=''):
    x = Conv2D(outdim, (3,3), strides=(1, 1), dilation_rate=dilated, kernel_initializer='he_normal',
               padding="same" )(input)
    if is_batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x
def DAC(x, outdim):
    x = Conv2D(outdim, (3,3), strides=(1, 1), kernel_initializer='he_normal',
               padding="same" )(x)
    x1 =Conv2D(outdim, (3,3), strides=(1, 1), dilation_rate=1, kernel_initializer='he_normal',
               padding="same" )(x)
    x1= Activation('relu')(x1)

    x2 =Conv2D(outdim, (3,3), strides=(1, 1), dilation_rate=3, kernel_initializer='he_normal',
               padding="same" )(x)
    x2=Conv2D(outdim, (1,1), strides=(1, 1), dilation_rate=1, kernel_initializer='he_normal',
               padding="same" )(x2)
    x2= Activation('relu')(x2)

    x3 =Conv2D(outdim, (3,3), strides=(1, 1), dilation_rate=1, kernel_initializer='he_normal',
               padding="same" )(x)
    x3 =Conv2D(outdim, (3,3), strides=(1, 1), dilation_rate=3, kernel_initializer='he_normal',
               padding="same" )(x3)
    x3=Conv2D(outdim, (1,1), strides=(1, 1), dilation_rate=1, kernel_initializer='he_normal',
               padding="same" )(x3)
    x3= Activation('relu')(x3)

    x4 =Conv2D(outdim, (3,3), strides=(1, 1), dilation_rate=1, kernel_initializer='he_normal',
               padding="same" )(x)
    x4 =Conv2D(outdim, (3,3), strides=(1, 1), dilation_rate=3, kernel_initializer='he_normal',
               padding="same" )(x4)
    x4 =Conv2D(outdim, (3,3), strides=(1, 1), dilation_rate=5, kernel_initializer='he_normal',
               padding="same" )(x4)
    x4=Conv2D(outdim, (1,1), strides=(1, 1), dilation_rate=1, kernel_initializer='he_normal',
               padding="same" )(x4)
    x4= Activation('relu')(x4)
    x=add([x, x1 , x2 , x3 , x4 ])
    return x

def CONV2D(x, filter_num, kernel_size, activation='relu', **kwargs):
    x = Conv2D(filter_num, kernel_size, padding='same',kernel_initializer='he_normal')(x) # , 
    x = BatchNormalization(axis=3)(x)
    if activation=='relu': 
        x = Activation('relu', **kwargs)(x)
    elif activation=='sigmoid': 
        x = Activation('sigmoid', **kwargs)(x)
    else:
        x = Activation('softmax', **kwargs)(x)
    return x

def BG_CNN(shape, classes=1):
    inputs = Input(shape)
    conv0 = BatchNormalization()(inputs)

    scale_img_2 = AveragePooling2D(pool_size=(2, 2))(inputs)
    scale_img_3 = AveragePooling2D(pool_size=(2, 2))(scale_img_2)
    scale_img_4 = AveragePooling2D(pool_size=(2, 2))(scale_img_3)

    conv0=DAC(conv0,32)
    conv0 = basic_Block(conv0, 32,with_conv_shortcut=True, name='down11')
    conv0 = basic_Block(conv0, 32, name='down11_2') 
    conv1 = basic_Block(conv0, 32, name='down11_3') 
    edge1 = Subtract()([conv0, conv1])
    conv1 = CONV2D(Concatenate(axis=3)([conv1, edge1]), 32, (3,3))
    pool1 = Conv2D(32, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal',name='pool1')(conv1)
    pool1=BatchNormalization()(pool1)


    conv0=DAC(scale_img_2,64)
    conv0 = Concatenate(axis=3)([conv0, pool1])
    conv0 = basic_Block(conv0, 64,with_conv_shortcut=True, name='down21')
    conv0 = basic_Block(conv0, 64, name='down22')
    conv0 = basic_Block(conv0, 64, name='down23')
    conv2 = basic_Block(conv0, 64, name='down24')
    edge2 = Subtract()([ conv0,conv2])
    conv2 = CONV2D(Concatenate(axis=3)([conv2, edge2]), 64, (3,3))
    pool2 = Conv2D(64, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal',name='pool2')(conv2)
    pool2=BatchNormalization()(pool2)   


    conv0=DAC(scale_img_3,128)
    conv0 = Concatenate(axis=3)([conv0, pool2])
    conv0 = basic_Block(conv0, 128,  with_conv_shortcut=True, name='down31')
    conv0 = basic_Block(conv0, 128,  name='down32')
    conv0 = basic_Block(conv0, 128,  name='down33')
    conv3 = basic_Block(conv0, 128,  name='down34')
    edge3 = Subtract()([conv0, conv3])
    conv3 = CONV2D(Concatenate(axis=3)([conv3, edge3]), 128, (3,3))
    pool3 = Conv2D(128, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal',name='pool3')(conv3)
    pool3=BatchNormalization()(pool3)


    conv0=DAC(scale_img_4,256)
    conv0 = Concatenate(axis=3)([conv0, pool3])
    conv0 = basic_Block(conv0, 256,  with_conv_shortcut=True, name='down41')
    conv0 = basic_Block(conv0, 256,  name='down42')
    conv0 = basic_Block(conv0, 256,  name='down43')
    conv0 = basic_Block(conv0, 256,  name='down44')
    conv0 = basic_Block(conv0, 256,  name='down45')
    conv4 = basic_Block(conv0, 256,  name='down46')
    edge4 = Subtract()([conv0, conv4])
    conv4 = CONV2D(Concatenate(axis=3)([conv4, edge4]), 256, (3,3))
    pool4 = Conv2D(256, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal',name='pool4')(conv4)
    pool4=BatchNormalization()(pool4)

    conv0 = basic_Block(pool4, 512, with_conv_shortcut=True, name='down51')
    conv0 = basic_Block(conv0, 512, name='down52')
    conv5 = basic_Block(conv0, 512, name='down53')
    edge5 = Subtract()([conv0, conv5]);
    conv5 = CONV2D(Concatenate(axis=3)([conv5, edge5]), 512, (3,3))


    up1 = UpSampling2D(size=(2, 2))(edge5)
    merg1 = Concatenate(axis=3)([up1, edge4])
    conv0 = basic_Block(merg1, 256, with_conv_shortcut=True)
    conv6 = basic_Block(conv0, 256);
    edge6 = Subtract()([conv0, conv6]);
    print(edge6.shape)
    up1 = UpSampling2D(size=(2, 2))(edge6)
    merg1 = Concatenate(axis=3)([up1, edge3])
    conv0 = basic_Block(merg1, 128, with_conv_shortcut=True)
    conv7 = basic_Block(conv0, 128);
    edge7 = Subtract()([conv0, conv7]);
    print(edge7.shape)
    up1 = UpSampling2D(size=(2, 2))(edge7)
    merg1 = Concatenate(axis=3)([up1, edge2])
    conv0 = basic_Block(merg1, 64, with_conv_shortcut=True)
    conv8 = basic_Block(conv0, 64);
    edge8 = Subtract()([conv0, conv8]);
    print(edge8.shape)
    up1 = UpSampling2D(size=(2, 2))(edge8)
    merg1 = Concatenate(axis=3)([up1, edge1])
    conv0 = basic_Block(merg1, 32, with_conv_shortcut=True)
    conv9 = basic_Block(conv0, 32);
    edge9 = Subtract()([conv0, conv9]);
    Boundary = CONV2D(edge9, classes, (1, 1), activation='sigmoid')


    up1 = UpSampling2D(size=(2, 2))(conv5)
    merg1 = Concatenate(axis=3)([up1, conv4, edge4, edge6])
    conv0 = basic_Block(merg1, 256, with_conv_shortcut=True)
    conv6 = basic_Block(conv0, 256);
    edge6 = Subtract()([conv0, conv6]);
    conv6 = CONV2D(Concatenate(axis=3)([conv6, edge6]), 256, (3,3));
    
    up1 = UpSampling2D(size=(2, 2))(conv6)
    merg1 = Concatenate(axis=3)([up1, conv3, edge3, edge7])
    conv0 = basic_Block(merg1, 128, with_conv_shortcut=True)
    conv7 = basic_Block(conv0, 128);
    edge7 = Subtract()([conv0, conv7]);
    conv7 = CONV2D(Concatenate(axis=3)([conv7, edge7]), 128, (3,3));

    up1 = UpSampling2D(size=(2, 2))(conv7)
    merg1 = Concatenate(axis=3)([up1, conv2, edge2, edge8])
    conv0 =basic_Block(merg1, 64, with_conv_shortcut=True)
    conv8 = basic_Block(conv0, 64);
    edge8 = Subtract()([conv0, conv8]);
    conv8 = CONV2D(Concatenate(axis=3)([conv8, edge8]), 64, (3,3));

    up1 = UpSampling2D(size=(2, 2))(conv8)
    merg1 = Concatenate(axis=3)([up1, conv1, edge1, edge9])
    conv0 = basic_Block(merg1, 32, with_conv_shortcut=True)
    conv9 = basic_Block(conv0, 32);
    edge9 = Subtract()([conv0, conv9]);
    conv9 = CONV2D(Concatenate(axis=3)([conv9, edge9]), 32, (3,3));

    Object = CONV2D(conv9, classes, (1, 1), activation='sigmoid')

    model = Model(input=inputs, output=[Object, Boundary])
    model.summary()
    return model



