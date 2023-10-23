import os, sys, glob,ntpath, time, datetime
import random, shutil, cv2
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt
import imgaug as ia
from imgaug import augmenters as iaa
from typing import List, Tuple, Dict
from keras.losses import mean_squared_error
from keras.optimizers import Adam, SGD, RMSprop
from keras.metrics import binary_accuracy, binary_crossentropy, mean_squared_error, mean_absolute_error
from keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau
from keras.callbacks import LearningRateScheduler, EarlyStopping, TensorBoard, CSVLogger
from keras.models import Model, load_model
from keras.applications import imagenet_utils
from keras import backend as K
import os
from PIL import Image as pil_image
import Paper3Net
import BasicFunc
from scipy.ndimage import distance_transform_edt as distance

global Train_Size
Train_Size = 512

np.random.seed(10000)
random.seed(20000000)






def dice_loss(y_true, y_pred):
    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
    union = tf.reduce_sum(tf.square(y_true)) + tf.reduce_sum(tf.square(y_pred))
    return - tf.log((2. * intersection + K.epsilon()) / (union + K.epsilon()))

def dice_loss1(y_true, y_pred):
    y_true2 = y_true[0]
    y_pred2 = y_pred[0]
    return dice_loss(y_true2, y_pred2)

def dice_loss2(y_true, y_pred):
    y_true1 = y_true[1]
    y_pred1 = y_pred[1]
    return dice_loss(y_true1, y_pred1)

def boundaryanddice(y_true, y_pred):
    return 0.6 * dice_loss1(y_true, y_pred) + 0.4 * dice_loss2(y_true, y_pred)

iaa_augmentation = iaa.SomeOf((0, 2), [
    iaa.Fliplr(0.5),
    iaa.Affine(scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
               rotate=(-10, 10)),  # , shear=(-5, 5)

])


def batch_augmentation(data_batch, iaa_augmentation):
    seq_det = iaa_augmentation.to_deterministic()
    while True:
        batch_x, [batch_y, batch_z] = next(data_batch)
        batch_x = seq_det.augment_images(batch_x)
        batch_y = seq_det.augment_images(batch_y)
        batch_z = seq_det.augment_images(batch_z)
        yield (batch_x.astype('float32'), [batch_y.astype('float32'), batch_z.astype('float32')])


def load_img(path, grayscale=False, target_size=None):
    """Loads an image into PIL format."""
    img = pil_image.open(path)
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            img = img.resize(width_height_tuple, pil_image.NEAREST)
    img = np.asarray(img, dtype='uint8')  #
    return img  # (512, 256, 3)


def save_img(img, file_name, scale=True):
    img = np.asarray(img, dtype='float32')
    if scale:
        img = img + max(-np.min(img), 0)
        x_max = np.max(img)
        if x_max != 0:
            img /= x_max
        img *= 255

    if len(img.shape) == 3:  # RGB
        pimg = pil_image.fromarray(img.astype('uint8'), 'RGB')
    elif len(img.shape) == 2:  # grayscale
        pimg = pil_image.fromarray(img.astype('uint8'), 'L')
    pimg.save(file_name)


def image_generator(batch_files, batch_size, train_size):
    while True:  # if random_state:
        random.shuffle(batch_files)

        img_list = []
        label1_list = []
        label2_list = []
        for batch_idx, batch_file in enumerate(batch_files):
            img = load_img(batch_file[0], grayscale=False, target_size=train_size)
            msk = load_img(batch_file[1], grayscale=True, target_size=train_size)
            msk1 = np.zeros_like(msk)
            msk2 = np.zeros_like(msk)
            msk3 = np.zeros_like(msk)
            msk4 = np.zeros_like(msk)
            msk1[msk == 1] = 1
            msk2[msk == 2] = 1
            msk3[msk == 3] = 1
            msk4[msk == 4] = 1

            kernel = np.ones((3, 3), np.uint8)
            msk1a = cv2.erode(np.uint8(msk1), kernel)
            msk2a = cv2.erode(np.uint8(msk2), kernel)
            msk3a = cv2.erode(np.uint8(msk3), kernel)
            msk4a = cv2.erode(np.uint8(msk4), kernel)

            edge1 = msk1 - msk1a
            edge2 = msk2 - msk2a
            edge3 = msk3 - msk3a
            edge4 = msk4 - msk4a

            edge1[edge1 > 0] = 1
            edge2[edge2 > 0] = 1
            edge3[edge3 > 0] = 1
            edge4[edge4 > 0] = 1

            if len(img.shape) == 2: img = np.expand_dims(img, axis=-1)
            if len(msk1.shape) == 2: msk1 = np.expand_dims(msk1, axis=-1)
            if len(msk2.shape) == 2: msk2 = np.expand_dims(msk2, axis=-1)
            if len(msk3.shape) == 2: msk3 = np.expand_dims(msk3, axis=-1)
            if len(msk4.shape) == 2: msk4 = np.expand_dims(msk4, axis=-1)

            if len(edge1.shape) == 2: edge1 = np.expand_dims(edge1, axis=-1)
            if len(edge2.shape) == 2: edge2 = np.expand_dims(edge2, axis=-1)
            if len(edge3.shape) == 2: edge3 = np.expand_dims(edge3, axis=-1)
            if len(edge4.shape) == 2: edge4 = np.expand_dims(edge4, axis=-1)

            mask = np.dstack([msk1, msk2, msk3, msk4])
            edge = np.dstack([edge1, edge2, edge3, edge4])

            img_list.append([img])
            label1_list.append([mask])
            label2_list.append([edge])
            if len(img_list) >= batch_size:
                x = np.vstack(img_list)
                y = np.vstack(label1_list)
                z = np.vstack(label2_list)
                yield x, [y, z]
                img_list = []
                label1_list = []
                label2_list = []


def step_decay(epoch):
    res = 1.0e-1
    if (epoch > 20) and (epoch <= 40):
        res = 1.0e-2
    elif (epoch > 40) and (epoch <= 60):
        res = 1.0e-3
    elif (epoch > 60) and (epoch <= 80):
        res = 1.0e-4
    elif (epoch > 80):
        res = 1.0e-5
    print("learnrate: ", res, " epoch: ", epoch)
    return res


def main(pathLists):
    train_rate = 0.9
    num = len(pathLists[0]) - int(len(pathLists[0]) * train_rate)
    imgPath = pathLists[0][num * 6:num * 7]
    mskPath = pathLists[1][num * 6:num * 7]
    pathLists[0] = [item for item in pathLists[0] if item not in imgPath]
    pathLists[1] = [item for item in pathLists[1] if item not in mskPath]
    trainPathList = [sorted(pathLists[0]), sorted(pathLists[1])]
    [train_files, valid_files] = BasicFunc.imgDumpAndDivideFromDirectory_seg(trainPathList[0], trainPathList[1], 0.8, debug=False)
    batch_size = 2
    epoch_count = 100
    train_gen = image_generator(train_files, batch_size, (Train_Size, Train_Size))
    valid_gen = image_generator(valid_files, batch_size, (Train_Size, Train_Size))

    train_gen = batch_augmentation(train_gen, iaa_augmentation)
    valid_gen = batch_augmentation(valid_gen, iaa_augmentation)

    models = [Paper3Net.BG_CNN]
    if True:
        base_folder = BasicFunc.dirMake(result_path[:8] + "BGCNN/Fold03")
        for item in models[0:1]:
            model_folder = BasicFunc.dirMake(base_folder + "/" + item.__name__)
            model_debug_fp = BasicFunc.dirMake(base_folder + "/" + item.__name__ + "/debug")
            model = item(shape=(Train_Size, Train_Size, 3), classes=4)
            model.compile(optimizer=SGD(lr=1.0e-4, momentum=0.90, nesterov=True), loss=boundaryanddice,
                          metrics=[boundaryanddice])

            if 1:
                cb_lr = LearningRateScheduler(step_decay)
                cb_tb = TensorBoard(log_dir=model_folder, write_images=False)
                cb_es = EarlyStopping(monitor='val_loss', min_delta=1.0e-7, patience=20, verbose=2, mode='auto')
                cb_cp = ModelCheckpoint(
                    model_folder + '/' + item.__name__ + "_Epoch" + str(epoch_count) + "_Batch" + str(
                        batch_size) + "_imSize" + str(Train_Size) + "_img.hd5", monitor='val_loss', verbose=2,
                    save_best_only=True, save_weights_only=False)
                cb_lg = CSVLogger(os.path.join(model_folder,
                                               item.__name__ + "_Epoch" + str(epoch_count) + "_Batch" + str(
                                                   batch_size) + "_imSize" + str(Train_Size) + '_img.csv'))
                callbacks = [cb_cp, cb_es, cb_tb, cb_lr, cb_lg]

                model.fit_generator(train_gen, len(train_files) / batch_size * 1., epoch_count,
                                    validation_data=valid_gen, nb_val_samples=len(valid_files) / batch_size * 1.,
                                    callbacks=callbacks)  #




# slice by slice to predict



CUDA_VISIBLE_DEVICES = 1

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.90)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    result_path = "results/debug"
    BasicFunc.dirMake(result_path)
    for file_path in glob.glob(result_path + "/*.*"): os.remove(file_path)

    num = 8000
    if 1:
        R0path = sorted(glob.glob(r"your path")[:num])
        R1path = sorted(glob.glob(r"your path")[:num])
        print(len(R0path), len(R1path))

        pathLists = [R0path, R1path]
        del R0path, R1path

        start = time.time()
        main(pathLists)
        print("\nruning time: ", time.time() - start)
        K.clear_session()
        sys.exit(1)
