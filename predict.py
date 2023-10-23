import os, sys, glob, time
import random,  cv2
import numpy as np
import tensorflow as tf
from imgaug import augmenters as iaa
from keras.optimizers import  SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler, EarlyStopping, TensorBoard, CSVLogger
from keras import backend as K

import os

from skimage.metrics import structural_similarity as SSIM
from scipy.spatial.distance import directed_hausdorff as HDistance
from PIL import Image as pil_image
import Paper3Net
import BasicFunc

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




def main(pathLists):

    imgPath = pathLists[0][:]
    mskPath = pathLists[1][:]
    batch_size = 2
    epoch_count = 100

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
                model_path = model_folder + '/' + item.__name__ + "_Epoch" + str(epoch_count) + "_Batch" + str(
                    batch_size) + "_imSize" + str(Train_Size) + "_img.hd5"
                model.load_weights(model_path)
                result_file = model_folder + '/' + "Predict_Epoch" + str(epoch_count) + "_Batch" + str(
                    batch_size) + "_imSize" + str(Train_Size) + '_' + item.__name__ + "_img.txt"

                predict_images(model, imgPath, mskPath, model_debug_fp, result_file)
                statistic_file = model_folder + '/' + "statistic_" + "Epoch" + str(epoch_count) + "_Batch" + str(
                    batch_size) + "_imSize" + str(Train_Size) + '_' + item.__name__ + "_img.txt"
                BasicFunc.txtStatisticEvaluation_seg1(result_file.replace('.txt', '_1.txt'),
                                                      statistic_file.replace('.txt', '_1.txt'))
                BasicFunc.txtStatisticEvaluation_seg1(result_file.replace('.txt', '_2.txt'),
                                                      statistic_file.replace('.txt', '_2.txt'))
                BasicFunc.txtStatisticEvaluation_seg1(result_file.replace('.txt', '_3.txt'),
                                                      statistic_file.replace('.txt', '_3.txt'))
                BasicFunc.txtStatisticEvaluation_seg1(result_file.replace('.txt', '_4.txt'),
                                                      statistic_file.replace('.txt', '_4.txt'))
                K.clear_session()


def predict_images(model, img_path, msk_path, model_debug_fp, file_name="Predict.txt"):
    file_name = file_name.replace("\\", "/")
    thefile0 = open(file_name.replace('.txt', '_1.txt'), "w")
    thefile1 = open(file_name.replace('.txt', '_2.txt'), "w")
    thefile2 = open(file_name.replace('.txt', '_3.txt'), "w")
    thefile3 = open(file_name.replace('.txt', '_4.txt'), "w")
    for i in range(len(img_path)):
        img = load_img(img_path[i], grayscale=False, target_size=(Train_Size, Train_Size))
        msk = load_img(msk_path[i], grayscale=True)
        msk = np.array(np.uint8(msk))
        msk[msk > 4] = 0
        msk1 = np.zeros_like(msk, dtype=np.float32)
        msk2 = np.zeros_like(msk, dtype=np.float32)
        msk3 = np.zeros_like(msk, dtype=np.float32)
        msk4 = np.zeros_like(msk, dtype=np.float32)
        msk1[msk == 1] = 1
        msk2[msk == 2] = 1
        msk3[msk == 3] = 1
        msk4[msk == 4] = 1


        temp = img.reshape([1] + list(img.shape))

        p = model.predict(temp.astype(np.float32))
        m = np.squeeze(p[0])
        e = np.squeeze(p[1])

        mp1 = m[:, :, 0];
        mp2 = m[:, :, 1];
        mp3 = m[:, :, 2];
        mp4 = m[:, :, 3];
        ep1 = e[:, :, 0];
        ep2 = e[:, :, 1];
        ep3 = e[:, :, 2];
        ep4 = e[:, :, 3]

        # p1 = BasicFunc.imgIntensityNormalize(p1,lower=0., upper=1.)
        mp1 = cv2.resize(mp1, msk.shape[::-1])
        mp2 = cv2.resize(mp2, msk.shape[::-1])
        mp3 = cv2.resize(mp3, msk.shape[::-1])
        mp4 = cv2.resize(mp4, msk.shape[::-1])
        mp1[mp1 < 0.5] = 0;
        mp1[mp1 >= 0.5] = 1
        mp2[mp2 < 0.5] = 0;
        mp2[mp2 >= 0.5] = 1
        mp3[mp3 < 0.5] = 0;
        mp3[mp3 >= 0.5] = 1
        mp4[mp4 < 0.5] = 0;
        mp4[mp4 >= 0.5] = 1

        ep1 = cv2.resize(ep1, msk.shape[::-1])
        ep2 = cv2.resize(ep2, msk.shape[::-1])
        ep3 = cv2.resize(ep3, msk.shape[::-1])
        ep4 = cv2.resize(ep4, msk.shape[::-1])
        ep1[ep1 < 0.5] = 0;
        ep1[ep1 >= 0.5] = 1
        ep2[ep2 < 0.5] = 0;
        ep2[ep2 >= 0.5] = 1
        ep3[ep3 < 0.5] = 0;
        ep3[ep3 >= 0.5] = 1
        ep4[ep4 < 0.5] = 0;
        ep4[ep4 >= 0.5] = 1

        newName = model_debug_fp + '/' + img_path[i].split("\\")[-1][:-4] + '_edge.png'
        cv2.imwrite(newName, np.uint8(100 * ep1 + 150 * ep2 + 50 * ep3 + 200 * ep4))
        newName = model_debug_fp + '/' + img_path[i].split("\\")[-1][:-4] + '_edge1.png'
        cv2.imwrite(newName, np.uint8(100 * ep1))
        newName = model_debug_fp + '/' + img_path[i].split("\\")[-1][:-4] + '_edge2.png'
        cv2.imwrite(newName, np.uint8(150 * ep2))
        newName = model_debug_fp + '/' + img_path[i].split("\\")[-1][:-4] + '_edge3.png'
        cv2.imwrite(newName, np.uint8(50 * ep3))
        newName = model_debug_fp + '/' + img_path[i].split("\\")[-1][:-4] + '_edge4.png'
        cv2.imwrite(newName, np.uint8(200 * ep4))
        newName = model_debug_fp + '/' + img_path[i].split("\\")[-1][:-4] + '_pred1.png'
        cv2.imwrite(newName, np.uint8(100 * mp1))
        newName = model_debug_fp + '/' + img_path[i].split("\\")[-1][:-4] + '_pred2.png'
        cv2.imwrite(newName, np.uint8(150 * mp2))
        newName = model_debug_fp + '/' + img_path[i].split("\\")[-1][:-4] + '_pred3.png'
        cv2.imwrite(newName, np.uint8(50 * mp3))
        newName = model_debug_fp + '/' + img_path[i].split("\\")[-1][:-4] + '_pred4.png'
        cv2.imwrite(newName, np.uint8(200 * mp4))
        newName1 = model_debug_fp + '/' + img_path[i].split("\\")[-1][:-4] + '_pred.png'
        cv2.imwrite(newName1, np.uint8(51 * mp1 + 101 * mp2 + 153 * mp3 + 204 * mp4))

        def dsc_iou(mskk1, mskk2):
            intersection = np.sum(mskk1.flatten() * mskk2.flatten())
            dsc = (2. * intersection) / (np.sum(mskk1) + np.sum(mskk2) + 0.001)
            overlap = intersection / (np.sum(mskk1) + np.sum(mskk2) - intersection + 0.001)

            mskk0 = mskk1 + 1
            pk0 = mskk2 + 1
            mskk0[mskk0 == 4] = 0
            pk0[pk0 == 4] = 0
            tp = intersection
            tn = np.sum(mskk0.flatten() * pk0.flatten())
            fp = np.sum(mskk2) - tp
            fn = np.sum(mskk1) - tp
            Se = tp / (tp + fn + ((tp + fn) == 0))
            Sp = tn / (tn + fp + ((tn + fp) == 0))
            Acc = (Se + Sp) / 2.
            acccorrrct = (tp + tn) / (tp + fn + tn + fp)
            Precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * Precision * Se / (Precision + Se)
            numerator = (tp * tn - fp * fn)
            denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            mcc = numerator / (denominator + 0.00001)
            return dsc, overlap, Acc, Se, mcc, Precision, f1, acccorrrct, recall, Sp

        dsc0, iou0, Acc0, Se0, mcc0, Precision0, f10, acccorrrct0, recall0, Sp0 = dsc_iou(msk1, mp1)
        dsc1, iou1, Acc1, Se1, mcc1, Precision1, f11, acccorrrct1, recall1, Sp1 = dsc_iou(msk2, mp2)
        dsc2, iou2, Acc2, Se2, mcc2, Precision2, f12, acccorrrct2, recall2, Sp2 = dsc_iou(msk3, mp3)
        dsc3, iou3, Acc3, Se3, mcc3, Precision3, f13, acccorrrct3, recall3, Sp3 = dsc_iou(msk4, mp4)
        hd0 = max(HDistance(msk1, mp1)[0], HDistance(mp1, msk1)[0])
        hd1 = max(HDistance(msk2, mp2)[0], HDistance(mp2, msk2)[0])
        hd2 = max(HDistance(msk3, mp3)[0], HDistance(mp3, msk3)[0])
        hd3 = max(HDistance(msk4, mp4)[0], HDistance(mp4, msk4)[0])

        ssim0 = SSIM(msk1, mp1)
        ssim1 = SSIM(msk2, mp2)
        ssim2 = SSIM(msk3, mp3)
        ssim3 = SSIM(msk4, mp4)
        img_type = img_path[i].split("\\")
        thefile0.writelines("%s\n" % str(
            [round(dsc0, 5), round(iou0, 5), round(Acc0, 5), round(acccorrrct0, 5), round(Se0, 5), round(mcc0, 5),
             round(Precision0, 5), round(f10, 5), round(hd0, 5), round(ssim0, 5), round(recall0, 5), round(Sp0, 5),
             img_type[-1]]))  #
        thefile1.writelines("%s\n" % str(
            [round(dsc1, 5), round(iou1, 5), round(Acc1, 5), round(acccorrrct1, 5), round(Se1, 5), round(mcc1, 5),
             round(Precision1, 5), round(f11, 5), round(hd1, 5), round(ssim1, 5), round(recall1, 5), round(Sp1, 5),
             img_type[-1]]))  #
        thefile2.writelines("%s\n" % str(
            [round(dsc2, 5), round(iou2, 5), round(Acc2, 5), round(acccorrrct2, 5), round(Se2, 5), round(mcc2, 5),
             round(Precision2, 5), round(f12, 5), round(hd2, 5), round(ssim2, 5), round(recall2, 5), round(Sp2, 5),
             img_type[-1]]))  #
        thefile3.writelines("%s\n" % str(
            [round(dsc3, 5), round(iou3, 5), round(Acc3, 5), round(acccorrrct3, 5), round(Se3, 5), round(mcc3, 5),
             round(Precision3, 5), round(f13, 5), round(hd3, 5), round(ssim3, 5), round(recall3, 5), round(Sp3, 5),
             img_type[-1]]))  #

    thefile0.close()
    thefile1.close()
    thefile2.close()
    thefile3.close()


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
