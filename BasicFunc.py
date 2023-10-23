#https://github.com/neurodata/ndreg/blob/master/ndreg/ndreg.py
#https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
import os, sys, glob,ntpath, time, datetime
import random, shutil, cv2, math
import numpy as np
import pandas as pd
import SimpleITK as sitk
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa # image augmentation

from urllib.request              import urlopen
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters       import gaussian_filter

import skimage as ski # image processing
from skimage import transform as skiTransform # image processing
import warnings
from keras import backend as K




#-------------------------------------------------------------------------------------
#   image transformatin based on openCV for 2D RGB images 
#-------------------------------------------------------------------------------------

class XYRange:
    '''
    The chance and range of 2D image transform for x and y axis, respectively
    '''
    def __init__(self, x_min, x_max, y_min, y_max, chance=0.5):
        self.chance = chance
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max


class XYZRange(XYRange):
    '''
    The chance and range of 2D image transform for x, y and z axis, respectively
    '''
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max, chance=0.5):
        super(XYZRange, self).__init__(x_min, x_max, y_min, y_max, chance)
        self.z_min = z_min
        self.z_max = z_max


def random_scale_img2D(img, xy_range, lock_xy=False):
    '''
    resize a 2D image with the given chance and ranges of x and y axis
    '''
    # bigger than a given chance and then scale img
    if random.random() < xy_range.chance: return img
        
    scale_x = random.uniform(xy_range.x_min, xy_range.x_max)
    scale_y = random.uniform(xy_range.y_min, xy_range.y_max)
    if lock_xy: scale_y = scale_x
    
    # size_x * size_y * size_z ==> size_y * size_x * size_z 
    # when using cv2.imread() function
    old_y, old_x = img.shape[:2] # only 0 and 1 for 2D img
    new_x  = round(old_x * scale_x)
    new_y = round(old_y * scale_y)
    scaled_img = cv2.resize(img, (new_x, new_y), interpolation=cv2.INTER_NEAREST)
    
    if new_x < old_x:
        extend_left = round((old_x - new_x) / 2.)
        extend_right = old_x - extend_left - new_x
        # 在 [top, bottom, left, right]区间内按指定方式填充图像左右边界
        scaled_img = cv2.copyMakeBorder(scaled_img, 0, 0, extend_left, extend_right, borderType=cv2.BORDER_CONSTANT)
        new_x = old_x
    
    if new_y < old_y:
        extend_top = round((old_y - new_y) / 2.)
        extend_bottom = old_y - extend_top - new_y
        scaled_img = cv2.copyMakeBorder(scaled_img, extend_top, extend_bottom, 0, 0,  borderType=cv2.BORDER_CONSTANT)
        new_y = old_y

    start_x = round((new_x - old_x) / 2.)
    start_y = round((new_y - old_y) / 2.)
    new_img = scaled_img[start_y: start_y + old_y, start_x: start_x + old_x]
    return new_img


def random_translate_img2D(img, xy_range, border_mode="constant"):
    '''
    translate 2D image with the given chance and ranges of x and y axis
    '''

    # bigger than a given chance and then translate img
    if random.random() < xy_range.chance: return img

    old_y, old_x = img.shape[:2]

    translate_x  = random.randint(xy_range.x_min, xy_range.x_max)
    translate_y  = random.randint(xy_range.y_min, xy_range.y_max)
    trans_matrix = np.array([[1, 0, translate_x], [0, 1, translate_y]]).astype(float) # 2D 图像平移操作
    border_const = cv2.BORDER_REFLECT if border_mode == "reflect" else cv2.BORDER_CONSTANT

    img = cv2.warpAffine(img, trans_matrix, (old_x, old_y), borderMode=border_const)
    return img


def random_rotate_img2D(img, chance, min_angle, max_angle):
    '''
    rotate 2D image with the given chance and angle in degrees
    '''

    # bigger than a given chance and then rotate img
    if random.random() < chance: return img

    size_x,size_y = img.shape[:2]
    rotate_center = (size_x/2., size_y/2.)
    angle         = random.randint(min_angle, max_angle)    
    rot_matrix    = cv2.getRotationMatrix2D(rotate_center, angle, scale=1.0) # in degrees

    img = cv2.warpAffine(img, rot_matrix, dsize=img.shape[:2], borderMode=cv2.BORDER_CONSTANT)
    return img


def random_flip_img2D(img, horizontal_chance=0., vertical_chance=0.):
    '''
    flip 2D image with the given chances of x and y axis, respectively
    '''
    flip_x = False
    flip_y = False

    # bigger than the given chance and then flip image
    if random.random() < horizontal_chance: flip_x = True
    if random.random() < vertical_chance:   flip_y = True
    if (not flip_x) and (not flip_y): return img

    flip_axis = 1
    if flip_y:
        flip_axis = -1 if flip_x else 0

    # 0 = X axis, 1 = Y axis,  -1 = both
    img_flip = cv2.flip(img, flip_axis)
    return img_flip


def random_gaussian_noise_img2D(img, noise_level, chance=0.5): #  "noise_level: 0-1"
    '''
    add gaussian noise to img with the given chance and noise level (-0.15~0.15)
    '''
    if abs(noise_level) > 0.15: 
        warnings.warn("noise_level should be in the range of -0.15~0.15")

    if random.random() < chance: return img
    # noise mean and std is 0 and 1, respectively
    random_value = np.random.normal(0.,1.,img.shape)
    noise_coefficient = random_value*noise_level

    noiseImg = img.astype(np.float)*(1 + noise_coefficient)
    [inMin, inMax] = limitOfDataType(img.dtype)

    noiseImg[noiseImg>inMax] = inMax
    noiseImg[noiseImg<inMin] = inMin
    noiseImg = noiseImg.astype(img.dtype)
    return noiseImg



# 只适合单通道图像变换
def elastic_transform_img2D(img, alpha, sigma, random_state=None):
    '''
    Map the input img to new coordinates by interpolation
    '''
    shape = img.shape
    if random_state is None: random_state = np.random.RandomState(1301)

    dx = gaussian_filter((random_state.rand(*shape) * 2. - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2. - 1), sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    elastic_grid = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)) # tuple

    return map_coordinates(img, elastic_grid, order=1).reshape(shape) 




#-------------------------------------------------------------------------------------
#   Basic functions to check the properities of parameters
#-------------------------------------------------------------------------------------

# degree to radian
Degree2Radian = (np.pi/180.)

Np2SitkDataTypes = { 'uint8'  : sitk.sitkUInt8,
                     'uint16' : sitk.sitkUInt16,
                     'uint32' : sitk.sitkUInt32,
                     'float32': sitk.sitkFloat32,
                     'uint64' : sitk.sitkUInt64}


Sitk2NpDataTypes = { sitk.sitkUInt8  : np.uint8,
                     sitk.sitkUInt16 : np.uint16,
                     sitk.sitkUInt32 : np.uint32,
                     sitk.sitkInt8   : np.int8,
                     sitk.sitkInt16  : np.int16,
                     sitk.sitkInt32  : np.int32,
                     sitk.sitkFloat32: np.float,
                     sitk.sitkFloat64: np.float64,}




def isIterable(variable):
    """
    Returns True if variable is a list, tuple or any other iterable object
    """
    return hasattr(variable, '__iter__')


def isNumber(variable):
    """
    Returns True if varible is is a number
    """
    return isinstance(variable, (int, float))


def isInteger(n, epsilon=1e-4):
    """
    Returns True if n is integer within error epsilon
    """
    # return ((n - int(n)) < epsilon)
    return type(n) and isinstance(n, int)


def isZero(n, epsilon=1e-4):
    """
    Return True if n is zero
    """
    if isNumber(n):
        return ((n-int(0.0)) < epsilon)
    else:
        return False


def INT(variable):
    """
    Return a integer object
    """
    if isNumber(variable):
        return int(variable+0.5)
    else:
        raise Exception("variable is not a number")


def isString(variable):
    """
    Returns True if varible is is a string
    """
    # try:
    #     type(variable)==str
    # except TypeError:
    #     return False
    return isinstance(variable, str)


def isNdarray(variable):
    """
    Returns True if variable is a np.ndarray object
    """
    return (type(variable)==np.ndarray)


def is2DOr3Dimg(img):
    """
    Return Trun if variable is a 2D or 3D image
    """
    if isNdarray(img):
        return (len(img.shape) in [2,3])
    else:
        return False


def isItkImg(img):
    """
    Returns True if image is a sitk.SimpleITK.Image object
    """
    return type(img)==sitk.SimpleITK.Image


def is2DOr3DItkImg(img):
    """
    Return True if image is a 2D or 3D SimpleITK.Image object
    """
    if isItkImg(img):
        return (img.GetDimension() in [2,3])
    else:
        return False


def isLIST(lst):
    """
    Return True if variable is a list or tuple
    """
    return (type(lst) in [list, tuple])


def isEmptyLIST(lst):
    """
    Return True if list or tuple is empty
    """
    if isLIST(lst):
        return isZero(len(lst))
    else:
        return False


def isLISTOfString(lst):
    """
    Check if variable is a list or tuple of numbers
    """
    if not isEmptyLIST(lst):
        return all([isString(item) for item in lst])
    else:
        return False


def isLISTOfNumber(lst):
    """
    Check if variable is a list or tuple of numbers
    """
    if not isEmptyLIST(lst):
        return all([isNumber(item) for item in lst])
    else:
        return False


def isLISTOfInteger(lst):
    """
    Check if variable is a list or tuple of intergers
    """
    if isLISTOfNumber(lst):
        return all([isInteger(item) for item in lst])
    else:
        return False


def hasZeroInLISTOfNumber(lst):
    """
    Check if zero exists in list or tuple of number
    """
    if isLISTOfNumber(lst):
        return any([isZero(item) for item in lst])
    else:
        return False


def hasZeroInLISTOfInteger(lst):
    """
    Check if zero exists in list or tuple of integer
    """
    if isLISTOfInteger(lst):
        return any([isZero(item) for item in lst])
    else:
        return False


def isRange(Rang):
    """
    Return True if variable is a effective range [lower, upper]
    """
    if isLISTOfNumber(Rang):
        if (len(Rang)==2):
            return (Rang[0]<Rang[1])
    else:
        return False


def isInRange(variable, Rang):
    """
    Retrun True if variable is in the range [lower, upper]
    """
    if isNumber(variable):
        if isRange(Rang):
            return ((Rang[0] <= variable) and (variable < Rang[1]))
        else:
            return False
    else:
        return False


def isSubrangeOfRange(range1,range2):
    """
    Return True if range1 is a subrange of rang2
    """
    if isRange(range1):
        if isRange(range2):
            return ((range2[0]<=range1[0]) and (range1[1]<=range2[1]))
        else:
            return False
    else:
        return False



#============================================================================


def pathFormat(filePath):
    """
    Returns file path using backslash ("/") such as "D:/Pitt/results/Glaucoma/"
    """
    if type(filePath) != str or dirPath == "":
        raise Exception("please provide a valid file path")
    target = tmp1.replace("\\", "/")
    return target


def dirMake(dirPath):
    """
    Returns directory path such as "D:/Pitt/results/Glaucoma/"
    """
    if type(dirPath) != str or dirPath == "":
        raise Exception("please provide a valid directory path")
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
        return os.path.normpath(dirPath)
    else:
        return dirPath


def getSuperFolderName(path):
    '''
    return the super folder name of a given file path

    example: 
        "folder" = getSuperFolderName("c:/folder/test.png")
    '''
    tmp=(path.replace("\\", "/")).split("/")
    if len(tmp)==1:
        raise Exception("path must contain a valid folder")
    else:
        SuperFolderName=tmp[-2]
    return SuperFolderName


def txtStatisticSameLine(inFile, element, outFile):
    """
    statistically the number of the same element in txt file
    """
    if not isString(inFile):
        raise Exception("file name must be a string")
    else:    
        if inFile.split(".")[-1]!="txt":
            raise Exception("input file name must be a .txt file")
    if not isString(element):
        raise Exception("element name must be a string")
    if not isString(outFile):
        raise Exception("output file name must be a string")
    fileOut = open(outFile, 'a')
    fileIn  = open(inFile, "r")
    content = fileIn.readlines() # read line by line
    content = [x.strip() for x in content] # delet "\n"
    num = 0
    for item in content:
        if item == element:
            num +=1
    fileOut.writelines("%-15s %-15s\n" % (element+": ", num))
    fileOut.close()
    fileIn.close()


def txtStatisticEvaluation(inFile, outFile):
    """
    statistically the number of the same element in txt file
    """
    if not isString(inFile):
        raise Exception("file name must be a string")
    else:    
        if inFile.split(".")[-1]!="txt":
            raise Exception("input file name must be a .txt file")
    if not isString(outFile):
        raise Exception("output file name must be a string")
    fileOut = open(outFile, 'a')
    fileIn  = open(inFile, "r")

    content = fileIn.readlines() # read line by line
    content = [x.strip() for x in content] # delet "\n"

    line = content[0].split(":")[0]
    classes = len(line.split(","))-1
    if int(math.sqrt(len(content))) !=classes or classes<0.:
        raise Exception("Please provide a valid file format")

    matrix = np.zeros((classes,classes))

    for i, item in enumerate(content):
        num = int(item.split(":")[-1])
        matrix[(i//classes),(i%classes)]=num

    pre=np.zeros(classes) # precision
    rec=np.zeros(classes) # recall
    f1 =np.zeros(classes) # f1 score
    
    tmp=0
    for i in range(classes):
        pre[i] = round(matrix[i,i]/(np.sum(matrix[:,i]+int(np.sum(matrix[:,i]==0.)))), 4)
        rec[i] = round(matrix[i,i]/np.sum(matrix[i,:]+int(np.sum(matrix[i,:]==0.))), 4)
        f1[i]  = round(2*pre[i]*rec[i]/(pre[i]+rec[i] + int(pre[i]+rec[i]==0.) ), 4)
        tmp += matrix[i,i]
    acc=round(tmp/(np.sum(matrix) + int(np.sum(matrix)==0.)), 4)
    fileOut.writelines("\n precision:%s\n recall:%s\n F1-score:%s\n acc:%s\n total images:%s" % (pre, rec, f1, acc, int(np.sum(matrix))) )
    fileOut.close()
    fileIn.close()

#============================================================================


def imgFlipAlongX(img):
    """
    Flip a numpy.ndarray image along axis X
    
    Input image  : numpy.ndarray
    Output image : numpy.ndarray 
    """
    if not is2DOr3Dimg(img):
        raise Exception("image must be a 2D or 3D numpy.ndarray object")
    
    if len(img.shape)==2:
        img=img[::-1,:]
    else:
        img=img[::-1,:,:]
    return img


def imgFlipAlongY(img):
    """
    Flip a numpy.ndarray image along axis Y
    
    Input image  : numpy.ndarray
    Output image : numpy.ndarray 
    """
    if not is2DOr3Dimg(img):
        raise Exception("image must be a 2D or 3D numpy.ndarray object")
    if len(img.shape)==2:
        img=img[:,::-1]
    else:
        img=img[:,::-1,:]
    return img


def imgFlipAlongZ(img):
    """
    Flip a numpy.ndarray image along axis Z
    
    Input image  : numpy.ndarray
    Output image : numpy.ndarray 
    """
    if not isNdarray(img):
        raise Exception("image must be a numpy.ndarray object")
    else:
        if len(img.shape) != 3:
            raise Exception("image must be a 3D numpy.ndarray object")
    return img[:,:,::-1]


def imgRandomFlipAlongX(img,chance=0.5):
    """
    Randomly flip image along axis x, with a probability of chance
    """
    if not is2DOr3Dimg(img):
        raise Exception("image must be a 2D or 3D numpy.ndarray object")
    if chance<0. or chance>1.:
        raise Exception("probability must be in the range [0., 1.]")
    if random.random() > chance: img = imgFlipAlongX(img)
    return img


def imgRandomFlipAlongY(img,chance=0.5):
    """
    Randomly flip image along axis y, with a probability of chance
    """
    if not is2DOr3Dimg(img):
        raise Exception("image must be a 2D or 3D numpy.ndarray object")
    if chance<0. or chance>1.:
        raise Exception("probability must be in the range [0., 1.]")
    if random.random() > chance: img = imgFlipAlongY(img)
    return img


def imgRandomFlipAlongZ(img,chance=0.5):
    """
    Randomly flip image along axis z, with a probability of chance
    """
    if not isNdarray(img):
        raise Exception("image must be a numpy.ndarray object")
    else:
        if len(img.shape)!=3:
            raise Exception("image must be a 3D numpy.ndarray object")
    if chance<0. or chance>1.:
        raise Exception("probability must be in the range [0., 1.]")
    if random.random() > chance: img = imgFlipAlongZ(img)
    return img




#==============================================================================
#   # from https://github.com/vxy10/ImageAugmentation
#==============================================================================

# def imgGammaChange(img, gamma=1.0):
#     """
#     # Gamma adjustment
#     # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
#     """
#     if not is2DOr3Dimg(img):
#         raise Exception("image must be a 2D or 3D numpy.ndarray object")
#     invGamma = 1.0 / gamma
#     table = np.array([((i / np.max(img)) ** invGamma) * np.max(img) for i in np.arange(0, np.max(img)+1)]).astype("uint8")
#     # apply gamma correction using the lookup table
#     new_img = cv2.LUT(img, table)
#     return new_img

# def imgRandomGammaChange(img, xy_range):
#     """
#     xy_range(0.7,1.3,0.7,1.3,0.5)
#     # Gamma adjustment
#     # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
#     """
#     if not is2DOr3Dimg(img):
#         raise Exception("image must be a 2D or 3D numpy.ndarray object")
#     if random.random() < xy_range.chance: return img
#     gamma = random.uniform(xy_range.x_min, xy_range.x_max)
#     img = imgGammaChange(img, gamma)
#     return img

def imgCLAHE(img, clipNum, gridSize):
    """
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    """
    if not is2DOr3Dimg(img):
        raise Exception("image must be a 2D or 3D numpy.ndarray object")
    if not isNumber(clipNum) or clipNum<=0:
        raise Exception("clipNum must be a positive value")
    if not isInteger(gridSize) or gridSize <1:
        raise Exception("gridSize must be a integer value")
    #create a CLAHE object (Arguments are optional).
    # clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4,4))
    clahe = cv2.createCLAHE(clipLimit=clipNum, tileGridSize=gridSize)
    new_img = np.empty(img.shape)
    if len(img.shape)==2: 
        new_img = clahe.apply(img)
    else:
        for i in range(img.shape[-1]):
            new_img[:,:,i] = clahe.apply(img[:,:,i])
    return new_img

def imgRandomCLAHE(img, xy_range):
    '''
    xy_range(1.,3.,3.,8,0.5)
    '''
    if not is2DOr3Dimg(img):
        raise Exception("image must be a 2D or 3D numpy.ndarray object")
    if random.random() < xy_range.chance: return img
    clipNum = random.uniform(xy_range.x_min, xy_range.x_max)
    gridSize= tuple([random.randint(xy_range.y_min, xy_range.y_max)]*2)
    img = imgCLAHE(img, clipNum, gridSize)
    return img


def imgRandomShear(img, rangeShear):
    """
    # Random shear transform an image with a given range (i.e., rangeShear=10)
    """
    if not is2DOr3Dimg(img):
        raise Exception("image must be a 2D or 3D numpy.ndarray object")   
    if not isNumber(rangeShear) or rangeShear <0:
        raise Exception("rangeShear must be a positive value")

    pts1 = np.float32([[5,5],[20,5],[5,20]])              # original coordinate points
    pt1 = 5+rangeShear*np.random.uniform()-rangeShear/2
    pt2 = 20+rangeShear*np.random.uniform()-rangeShear/2
    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])        # transformed coordinate points

    Matrix = cv2.getAffineTransform(pts1,pts2)
    rows,cols,ch = img.shape
    img = cv2.warpAffine(img,Matrix,(cols,rows))
    return img

# from https://github.com/vxy10/ImageAugmentation
def imgBrightnessAugment(img, bfactor):
    """
    Augment an image by changing its brightness with a given 'bfactor' in (0,1)
    """
    if not is2DOr3Dimg(img):
        raise Exception("image must be a 2D or 3D numpy.ndarray object")   
    if not isNumber(bfactor) or (bfactor < 0):
        raise Exception("bfactor must be a positive value")
    img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)

    # bfactor = .25+np.random.uniform()
    img[:,:,2] = img[:,:,2]*bfactor
    img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    return img

def imgRandomBrightnessAugment(img, xy_range):
    """
    Augment an image by changing its brightness with a random value in 'xy_range'
    """
    if not is2DOr3Dimg(img):
        raise Exception("image must be a 2D or 3D numpy.ndarray object")

    if random.random() < xy_range.chance: return img
    bfactor = random.randint(xy_range.x_min, xy_range.x_max)
    img = imgBrightnessAugment(img, bfactor)
    return img



def imgContrastAugment(img):
    """
    Augment an image by changing its contrast
    """
    if not is2DOr3Dimg(img):
        raise Exception("image must be a 2D or 3D numpy.ndarray object")   

    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    img = cv2.cvtColor(yuv,cv2.COLOR_YUV2RGB)
    return img


#==============================================================================
#==============================================================================










def imgCentralCrop(img, size):
    """
    crop patch with a specific size from an image
    """
    if not is2DOr3Dimg(img): 
        raise Exception('Image must be a 2D or 3D numpy.ndarray object')
    if (not isLISTOfInteger(size)):
        raise Exception("Size must be a list of integers")
    else:
        if len(img.shape) != len(size):
            raise Exception("len(img.shape) != len(size)")

    if img.shape == size: return img
    if len(img.shape)==2: 
        img=img.reshape(list(img.shape[:2])+[1])
        size=list(size)+[1]

    # # first, place the input image into a big box
    inSize = img.shape
    bigShape = [max(size[i], inSize[i]) for i in range(len(img.shape))]
        
    bigImg   = np.zeros(bigShape).astype(img.dtype)
    start = [round((bigShape[i] - inSize[i])/2.) for i in range(len(img.shape))]
    bigImg[start[0]: start[0]+inSize[0], 
           start[1]: start[1]+inSize[1], 
           start[2]: start[2]+inSize[2]] = img

    # # second, crop a patch with given size
    croped = np.zeros(size).astype(img.dtype)
    start = [round((bigShape[i] - size[i])/2.) for i in range(len(img.shape))]
    croped = bigImg[start[0]: start[0]+size[0], 
                    start[1]: start[1]+size[1], 
                    start[2]: start[2]+size[2]]

    if len(img.shape)==3 and img.shape[-1]==1: croped = croped[:,:,0]
    return croped


def imgIntensityTruncate(img, dataType=np.float16):
    """
    Truncate image intensity to keep them in the range of limits of dataType
    """
    if not isNdarray(img):
        raise Exception("image must be a np.ndarray object")
    
    [inMin, inMax] = limitOfDataType(np.dtype(img.dtype))
    img[img>inMax] = inMax
    img[img<inMin] = inMin
    return img


def limitOfDataType(dataType):
    """
    Return the lower and upper bounds of a given data type
    """
    if type(dataType) != type(np.dtype(dataType)):
        raise Exception("data type must be a numpy.dtype")
    if np.dtype(dataType).kind not in ['u','i','f']:
        raise RuntimeError("Only for integer or floating point data type")
    
    if np.dtype(dataType).kind in ['u','i']:
        iin = np.iinfo(dataType)
    else:
        iin = np.finfo(dataType)
    iinMax = iin.max
    iinMin = iin.min
    return [iinMin, iinMax]


def imgIntensityNormalize(img,lower=0., upper=1.):
    """
    shift image intensity to the range [lower, upper]
    """
    if not isNdarray(img):
        raise Exception("image must be a numpy.ndarray object")
    if (not isNumber(lower)) or (not isNumber(upper)):
        raise Exception("intensity lower and upper must be numbers")
    elif lower >= upper:
        raise Exception("Error: lower >= upper")
    if np.min(img) == np.max(img):
        raise Exception("Error: np.min(img) == np.max(img)")

    [dtypeMin, dtypeMax] = limitOfDataType(img.dtype)

    if ((upper > dtypeMax) or (lower < dtypeMin)):
        raise Exception("intensity lower and upper must be in the range of %s" %img.dtype)
    img0 = img.astype(np.float)

    img0 = (img0 - np.min(img0))/(np.max(img0) - np.min(img0))
    img0 = img0*(upper - lower) + lower
    # sitkImg1 = sitk.RescaleIntensity(sitkImg, lower, upper)
    return img0.astype(img.dtype)


def imgSave(img, folderName="results/debug", fileName="img"):
    '''
    Save an image to a given folder using a given "name"  
    '''
    if not is2DOr3Dimg(img) and not is2DOr3DItkImg(img):
        raise Exception("image must be a 2D or 3D numpy.ndarray or SimpleITK.Image object")
    if  not isString(folderName) and not isString(fileName):
        raise Exception("file and image name must be strings")

    if not os.path.exists(folderName): os.makedirs(folderName)

    if is2DOr3Dimg(img):
        sitkImg = sitk.GetImageFromArray(img)        
        if len(img.shape)==3 and img.shape[2]==3:
            sitkImg = sitk.Cast(sitkImg, sitk.sitkUInt8)
            sitkImg = sitk.Compose(sitkImg[0,:,:], sitkImg[1,:,:], sitkImg[2,:,:])
            imgName = folderName+"/"+fileName+"."+(datetime.datetime.now()).strftime("%Y.%m.%d.%H.%M.%S.%f")+".jpg"
        elif len(img.shape)==2:
            sitkImg = sitk.Cast(sitkImg, sitk.sitkUInt8)
            imgName = folderName+"/"+fileName+"."+(datetime.datetime.now()).strftime("%Y.%m.%d.%H.%M.%S.%f")+".jpg"
        else:
            imgName = folderName+"/"+fileName+"."+(datetime.datetime.now()).strftime("%Y.%m.%d.%H.%M.%S.%f")+".mhd"
        img = sitkImg
    elif is2DOr3DItkImg(img): 
        if img.GetDimension()==2 and img.GetNumberOfComponentsPerPixel()==3:
            img = sitk.Cast(img, sitk.sitkVectorUInt8)
            imgName = folderName+"/"+fileName+"."+(datetime.datetime.now()).strftime("%Y.%m.%d.%H.%M.%S.%f")+".jpg"
        elif img.GetDimension()==2 and img.GetNumberOfComponentsPerPixel()==1:
            img = sitk.Cast(img, sitk.sitkVectorUInt8)
            imgName = folderName+"/"+fileName+"."+(datetime.datetime.now()).strftime("%Y.%m.%d.%H.%M.%S.%f")+".jpg"
        else:
            imgName = folderName+"/"+fileName+"."+(datetime.datetime.now()).strftime("%Y.%m.%d.%H.%M.%S.%f")+".mhd"
    sitk.WriteImage(img,imgName)


def imgShow(img, color="gray"):
    '''
    show an image of numpy.ndarray type
    '''
    if not is2DOr3Dimg(img):
        raise Exception("image must be a 2D or 3D numpy.ndarray object")
    if type(color)!=str:
        raise Exception("provide a valid colormap for pyplot")

    plt.ion()
    if len(img.shape)==2:
        plt.figure()
        plt.imshow(img, cmap=plt.get_cmap(color))
    else:
        plt.figure()
        plt.imshow(img[:,:, round((img.shape[2]-1)/2.)], cmap=plt.get_cmap(color))


def imgRepeat(img,n):
    if not is2DOr3Dimg(img):
        raise Exception("image must be a 2D or 3D numpy.ndarray object")
    if not isInteger(n):
        raise Exception("variable 'n' must be an integer")

    # repeat "img" for 'n' times
    # Img=np.zeros(list(img.shape)+[n])
    # for i in range(n): Img[:,:,i] = img
    Img = np.dstack([img]*n)
    return Img









def processImageMask(simg, smsk):
    '''
    process a given image and its mask for specific tasks
    '''
    pass



def dumpImgLabel(imgPath, mskPath=[], label=[1,0], debug=False):
    """
    Return a list of [[img0,label0],[img1,label1],...]
    """
    if isEmptyLIST(imgPath) or (not isLISTOfString(imgPath)): 
        raise Exception("image path must be a non-empty list of string")

    if not isEmptyLIST(mskPath):
        if not isLISTOfString(mskPath):
            raise Exception("mask path must be a list of string")
        else:
            if (len(imgPath) != len(mskPath)):
                raise Exception("len(imgPath) != len(mskPath)")

    imageLabelList=[]
    for idx, data in enumerate(imgPath):
        simg = sitk.ReadImage(data)

        if not isEmptyLIST(mskPath):
                smsk = sitk.ReadImage(mskPath[idx])
                # res = processImgMsk(simg,smsk)
                imageLabelList.append([simg,smsk,label])
        else:
            imageLabelList.append([simg,label])

        if debug:
            folderName = "results/debug"
            if not os.path.exists(folderName): os.makedirs(folderName)
            superFolder=getSuperFolderName(data)
            folderFile = open(folderName+"/"+superFolder+".txt","a")
            folderFile.writelines("%s\n" % str([superFolder[:3], label]))

    # random.shuffle(imageLabelList)
    return imageLabelList


def dumpImgLabelFromDirectory(imgPath, mskPath=[], label=[1,0], debug=False):
    """
    Return a path list of [[img0,label0],[img1,label1],...]
    """
    if isEmptyLIST(imgPath) or (not isLISTOfString(imgPath)): 
        raise Exception("image path must be a non-empty list of string")

    if not isEmptyLIST(mskPath):
        if not isLISTOfString(mskPath):
            raise Exception("mask path must be a list of string")
        else:
            if (len(imgPath) != len(mskPath)):
                raise Exception("len(imgPath) != len(mskPath)")

    imageLabelList=[]
    for idx, data in enumerate(imgPath):
        if not isEmptyLIST(mskPath):
            imageLabelList.append([imgPath[idx],mskPath[idx],label])
        else:
            imageLabelList.append([imgPath[idx],label])

        if debug:
            folderName = "results/debug"
            if not os.path.exists(folderName): os.makedirs(folderName)
            superFolder=getSuperFolderName(data)
            folderFile = open(folderName+"/"+superFolder+".txt","a")
            folderFile.writelines("%s\n" % str([superFolder[:3], label]))

    return imageLabelList


def createLabel(subNum, sumNum):
    """
    create a image label (e.g., [1,0,0,0])
    """
    if not isInteger(subNum) and (not isInteger(sumNum)):
        raise Exception("both subNum and sumNum must be integer")
    if (sumNum < 1): 
        raise Exception("Sum of all classes must be greater than 1")
    if subNum > (sumNum-1) or subNum < 0: 
        raise Exception("subNum must be in the range [0, sumNum-1]")

    Num   = [i for i in range(sumNum)]
    label = [int(item==subNum) for item in Num]
    return label 


def imgDumpAndDivide(imgPath, mskPath=[], labelIndex=(0,2), divideRate=0.8, debug=False):
    """
    read and label images, and then divide them into train and validation parts for deep learning
    """
    if divideRate >1. or divideRate <0.:
        raise Exception(" ratio must be in the range [0.,1.]")
    imgLabel = createLabel(labelIndex[0], labelIndex[1])
    # imgLabel = keras.utils.to_categorical(labelIndex[0], labelIndex[1])

    data = dumpImgLabel(imgPath, mskPath, imgLabel, debug)

    # for train and validation in cnn model
    trainPart = data[:int(len(data)*divideRate)]
    validPart = data[int(len(data)*divideRate):]
    return [trainPart, validPart]


def imgDumpAndDivideFromDirectory(imgPath, mskPath=[], labelIndex=(0,2), divideRate=0.8, debug=False):
    """
    read and label images, and then divide them into train and validation parts for deep learning
    """
    if divideRate >1. or divideRate <0.:
        raise Exception(" ratio must be in the range [0.,1.]")
    imgLabel = createLabel(labelIndex[0], labelIndex[1])
    data = dumpImgLabelFromDirectory(imgPath, mskPath, imgLabel, debug)

    # for train and validation in cnn model
    trainPart = data[:int(len(data)*divideRate)]
    validPart = data[int(len(data)*divideRate):]
    return [trainPart, validPart]



#============================================================================
# new add for 6 channels (big RGB + small RGB)
#============================================================================

def dumpBigImgLabelFromDirectory(imgPath, mskPath=[], label=[1,0], debug=False):
    """
    Return a path list of [[img0,label0],[img1,label1],...]
    """
    if isEmptyLIST(imgPath) or (not isLISTOfString(imgPath)): 
        raise Exception("image path must be a non-empty list of string")
    if isEmptyLIST(mskPath) or (not isLISTOfString(mskPath)): 
        raise Exception("image path must be a non-empty list of string")
    
    imgPath=sorted(imgPath)
    mskPath=sorted(mskPath)

    imageLabelList=[]
    for idx, data in enumerate(imgPath):
        imgName = (imgPath[idx].replace('\\','/')).split('/')[-1][:-len('fieldview_norm.jpg')]   # fieldview_norm.jpg
        mskName = (mskPath[idx].replace('\\','/')).split('/')[-1][:-len('disc_norm.jpg')]   # disc_norm.jpg
        if imgName != mskName: continue

        imageLabelList.append([imgPath[idx],mskPath[idx],label])

        if debug:
            folderName = "results/debug"
            if not os.path.exists(folderName): os.makedirs(folderName)
            superFolder=getSuperFolderName(data)
            folderFile = open(folderName+"/"+superFolder+".txt","a")
            folderFile.writelines("%s\n" % str([superFolder[:3], label]))

    return imageLabelList



def bigImgDumpAndDivideFromDirectory(imgPath, mskPath=[], labelIndex=(0,2), divideRate=0.8, debug=False):
    """
    read and label images, and then divide them into train and validation parts for deep learning
    """
    if divideRate >1. or divideRate <0.:
        raise Exception(" ratio must be in the range [0.,1.]")
    imgLabel = createLabel(labelIndex[0], labelIndex[1])
    data = dumpBigImgLabelFromDirectory(imgPath, mskPath, imgLabel, debug)

    # for train and validation in cnn model
    trainPart = data[:int(len(data)*divideRate)]
    validPart = data[int(len(data)*divideRate):]
    return [trainPart, validPart]

#============================================================================



def dumpImgLabel_seg(imgPath, mskPath, debug=False):
    """
    Return a list of [[img0,msk0],[img1,msk1],...]
    """
    if isEmptyLIST(imgPath) or (not isLISTOfString(imgPath)): 
        raise Exception("image path must be a non-empty list of string")
    if isEmptyLIST(mskPath) or (not isLISTOfString(mskPath)): 
        raise Exception("mask path must be a non-empty list of string")

    if (len(imgPath) != len(mskPath)):
        raise Exception("len(imgPath) != len(mskPath)")
    imgPath = sorted(imgPath)
    mskPath = sorted(mskPath)

    imageLabelList=[]
    for idx, data in enumerate(imgPath):
        simg = sitk.ReadImage(imgPath[idx])
        smsk = sitk.ReadImage(mskPath[idx])
        imageLabelList.append([simg,smsk])

        if debug:
            folderName = "results/debug"
            if not os.path.exists(folderName): os.makedirs(folderName)
            superFolder=getSuperFolderName(data)
            folderFile = open(folderName+"/"+superFolder+".txt","a")
            folderFile.writelines("%s\n" % str([ superFolder[:5], imgPath[idx], mskPath[idx] ]))

    # random.shuffle(imageLabelList)
    return imageLabelList


def dumpImgLabelFromDirectory_seg(imgPath, mskPath, debug=False):
    """
    Return a list of [[img0,msk0],[img1,msk1],...]
    """
    if isEmptyLIST(imgPath) or (not isLISTOfString(imgPath)): 
        raise Exception("image path must be a non-empty list of string")
    if isEmptyLIST(mskPath) or (not isLISTOfString(mskPath)): 
        raise Exception("mask path must be a non-empty list of string")

    if (len(imgPath) != len(mskPath)):
        raise Exception("len(imgPath) != len(mskPath)")
    imgPath = sorted(imgPath)
    mskPath = sorted(mskPath)

    imageLabelList=[]
    for idx, data in enumerate(imgPath):
        imageLabelList.append([imgPath[idx],mskPath[idx]])

        if debug:
            folderName = "results/debug"
            if not os.path.exists(folderName): os.makedirs(folderName)
            superFolder=getSuperFolderName(data)
            folderFile = open(folderName+"/"+superFolder+".txt","a")
            folderFile.writelines("%s\n" % str([ superFolder[:5], imgPath[idx], mskPath[idx] ]))

    return imageLabelList



def imgDumpAndDivide_seg(imgPath, mskPath, divideRate=0.8, debug=False):
    """
    read and label images, and then divide them into train and validation parts for deep learning
    """
    if divideRate >1. or divideRate <0.:
        raise Exception(" ratio must be in the range [0.,1.]")

    data = dumpImgLabel_seg(imgPath, mskPath, debug)

    # for train and validation in cnn model
    trainPart = data[:round(len(data)*divideRate)]
    validPart = data[round(len(data)*divideRate):]
    return [trainPart, validPart]


def imgDumpAndDivideFromDirectory_seg(imgPath, mskPath, divideRate=0.8, debug=False):
    """
    read and label images, and then divide them into train and validation parts for deep learning
    """
    if divideRate >1. or divideRate <0.:
        raise Exception(" ratio must be in the range [0.,1.]")
    data = dumpImgLabelFromDirectory_seg(imgPath, mskPath, debug)

    # for train and validation in cnn model
    trainPart = data[:round(len(data)*divideRate)]
    validPart = data[round(len(data)*divideRate):]
    return [trainPart, validPart]


#-------------------------------------------------------------------------
def dumpImgLabelFromDirectory_seg2(imgPath1, imgPath2, mskPath, debug=False):
    """
    Return a list of [[img01,img02, msk0],[img11, img12, msk1],...]
    img01 and img02 can be integrated to get desirabel images
    """
    if isEmptyLIST(imgPath1) or (not isLISTOfString(imgPath1)): 
        raise Exception("image path must be a non-empty list of string")
    if isEmptyLIST(imgPath2) or (not isLISTOfString(imgPath2)): 
        raise Exception("image path must be a non-empty list of string")        
    if isEmptyLIST(mskPath) or (not isLISTOfString(mskPath)): 
        raise Exception("mask path must be a non-empty list of string")

    if (len(imgPath1) != len(mskPath) or len(imgPath2) != len(mskPath)):
        raise Exception("len(imgPath) != len(mskPath)")
    imgPath1 = sorted(imgPath1)
    imgPath2 = sorted(imgPath2)
    mskPath = sorted(mskPath)

    imageLabelList=[]
    for idx, data in enumerate(imgPath1):
        imageLabelList.append([imgPath1[idx], imgPath2[idx], mskPath[idx]])

        if debug:
            folderName = "results/debug"
            if not os.path.exists(folderName): os.makedirs(folderName)
            superFolder=getSuperFolderName(data)
            folderFile = open(folderName+"/"+superFolder+".txt","a")
            folderFile.writelines("%s\n" % str([ superFolder[:5], imgPath1[idx], mskPath[idx] ]))

    return imageLabelList

def imgDumpAndDivideFromDirectory_seg2(imgPath1, imgPath2, mskPath, divideRate=0.8, debug=False):
    """
    read and label images, and then divide them into train and validation parts for deep learning
    """
    if divideRate >1. or divideRate <0.:
        raise Exception(" ratio must be in the range [0.,1.]")
    data = dumpImgLabelFromDirectory_seg2(imgPath1, imgPath2, mskPath, debug)

    # for train and validation in cnn model
    trainPart = data[:round(len(data)*divideRate)]
    validPart = data[round(len(data)*divideRate):]
    return [trainPart, validPart]

#-------------------------------------------------------------------------


def txtStatisticEvaluation_seg(inFile, outFile):
    """
    statistically the number of the same element in txt file
    """
    if not isString(inFile):
        raise Exception("file name must be a string")
    else:
        if inFile.split(".")[-1]!="txt":
            raise Exception("input file name must be a .txt file")
    if not isString(outFile):
        raise Exception("output file name must be a string")
    fileOut = open(outFile, 'w')
    fileIn  = open(inFile, "r")

    content = fileIn.readlines() # read line by line
    content = [x.strip() for x in content] # delet "\n"
    
    # line like such '[0.939, 0.885, 0.945, 0.892, 0.999, 'DIARETDB0 (1).tif']' in 'inFile'
    content0 = [float(x.split(",")[0][1:]) for x in content]
    img0 = np.array(content0)
    dsc_m  = round(np.mean(img0), 4)
    dsc_std= round(np.std(img0), 4)

    content1 = [float(x.split(",")[1]) for x in content]
    img1 = np.array(content1)
    IoU_m  = round(np.mean(img1), 4)
    IoU_std= round(np.std(img1), 4)

    content2 = [float(x.split(",")[2]) for x in content]
    img2 = np.array(content2)
    Acc_m  = round(np.mean(img2), 4)
    Acc_std= round(np.std(img2), 4)

    content3 = [float(x.split(",")[3]) for x in content]
    img3 = np.array(content3)
    Se_m = round(np.mean(img3), 4)
    Se_std= round(np.std(img3), 4)

    content4 = [float(x.split(",")[4]) for x in content]
    img4 = np.array(content4)
    Sp_m  = round(np.mean(img4), 4)
    Sp_std= round(np.std(img4), 4)

    fileOut.writelines( "\n dsc mean:%s\n dsc std:%s\n IoU mean:%s\n IoU std:%s\n Acc mean:%s\n Acc std:%s\n  Se mean:%s\n Se std:%s\n  Sp mean:%s\n Sp std:%s\n total images:%s" % (dsc_m, dsc_std, IoU_m, IoU_std, Acc_m, Acc_std, Se_m, Se_std, Sp_m, Sp_std, len(content1)))

    # fileOut.writelines("\n dsc mean:%s\n dsc std:%s\n dsc min:%s\n dsc max:%s\n overlap mean:%s\n overlap std:%s\n overlap min:%s\n overlap max:%s\n total images:%s" % (dsc_m, dsc_std, dsc_min, dsc_max,overlap_m, overlap_std, overlap_min,overlap_max, len(content1)))
    fileOut.close()
    fileIn.close()


def txtStatisticEvaluation_seg1(inFile, outFile):
    """
    statistically the number of the same element in txt file
    """
    if not isString(inFile):
        raise Exception("file name must be a string")
    else:
        if inFile.split(".")[-1]!="txt":
            raise Exception("input file name must be a .txt file")
    if not isString(outFile):
        raise Exception("output file name must be a string")
    fileOut = open(outFile, 'w')
    fileIn  = open(inFile, "r")

    content = fileIn.readlines() # read line by line
    content = [x.strip() for x in content] # delet "\n"
    
    # line like such '[0.939, 0.885, 0.945, 0.892, 0.999, 'DIARETDB0 (1).tif']' in 'inFile'
    m = len(content[0].split(",")) # the number of comma (',') in each line

    content0 = [float(x.split(",")[0][1:]) for x in content]
    img0 = np.array(content0)
    metric_mean=round(np.mean(img0), 4)
    metric_std=round(np.std(img0), 4)
    fileOut.writelines( "\n 0 metric mean+std :%s + %s" % (metric_mean, metric_std))
    
    # from 2nd to the (last-1)
    for ii in range(1, (m-1)): 
        content0 = [float(x.split(",")[ii]) for x in content]
        img0 = np.array(content0)
        metric_mean=round(np.mean(img0), 4)
        metric_std=round(np.std(img0), 4)
        fileOut.writelines( "\n %s metric mean+std :%s + %s" % (ii, metric_mean, metric_std))

    fileOut.writelines( "\n total images:%s" % len(content) )
    fileOut.close()
    fileIn.close()


def txtStatisticEvaluation_seg_thresh(inFile, outFile, thresh=0.85):
    """
    statistically the number of the same element in txt file
    """
    if not isString(inFile):
        raise Exception("file name must be a string")
    else:
        if inFile.split(".")[-1]!="txt":
            raise Exception("input file name must be a .txt file")
    if not isString(outFile):
        raise Exception("output file name must be a string")
    fileOut = open(outFile, 'a')
    fileIn  = open(inFile, "r")

    content = fileIn.readlines() # read line by line
    content = [x.strip() for x in content] # delet "\n"

    for item in content:
        tmp = float(item.split(",")[0][1:])
        # print(tmp, item)
        # input('......')
        if tmp>=thresh: fileOut.writelines("%s\n" % item)
    fileOut.close()
    fileIn.close()

#-------------------------------------------------------------------------------------
#                   Basic functions for CNNs
#-------------------------------------------------------------------------------------

def dumpForNetwork(simg, train_size):
    '''
    make a given image applicable to network
    last chance to process image before dumping into network
    '''
    if not is2DOr3DItkImg(simg):
        raise Exception("please provid a valid sitk.Image object")

    simg = sitkFunc.itkImgResample(simg,[], train_size)
    simg = sitk.Cast(simg, sitk.sitkVectorFloat32)
    # binMsk = sitk.BinaryThreshold(sitkMsk, 1, 5) # [1-5]之间为1，其他为0
    # dilMsk = sitk.Cast(sitk.BinaryDilate(binMsk,7), sitk.sitkFloat32)      
    # sitkImg = sitk.Multiply(sitkImg, dilMsk)

    # simg = sitk.RescaleIntensity(simg, 0,512)

    img = sitk.GetArrayFromImage(simg)
    # if len(img.shape) != simg.GetDimension():
    #     tmp = list(train_size) + [img.shape[-1]]
    # else:
    #     tmp = train_size
    # img = imgCentralCrop(img,tmp)
    return img


def image_generator(batch_files, batch_size, train_size, random_state=True, debug=False):
    '''
    dynamicly generate images for networks
    
    Input: 
        batch_files: list of image and label (i.e., [[img, lab],[img, lab],...])
        batch_size : an integer number
        train_size : image size that network needs 
    '''
    while True:
        # if random_state:
        random.shuffle(batch_files)

        img_list = []
        label_list = []
        for batch_idx, batch_file in enumerate(batch_files):
            simg = [batch_file[0],sitk.ReadImage(batch_file[0])][(batch_file[0]==str)]
            label= batch_file[-1]

            if len(train_size) != simg.GetDimension():
                raise Exception("please provide a valid 'train_size'")
            if len(batch_file)==3: 
                smsk=[batch_file[1],sitk.ReadImage(batch_file[1])][(batch_file[1]==str)]

            #------------------------------------
            # processing simg and smsk ...
            #------------------------------------

            if random_state:
                # simg = itkImgRandomAddGaussianNoise(simg, XYRange(0.,1.,0.,0.,0.5))
                simg = itkImgRandomFlipAlongX(simg, 0.5)
                simg = itkImgRandomFlipAlongY(simg, 0.5)
                # simg = itkImgRandomRotation(simg,   XYZRange(-30,30, -30,30, -30,30, 0.5))
                # simg = itkImgRandomZoom(simg, XYZRange(0.85,1.15,0.85,1.15,0.85,1.15,0.5),lock_xyz=True)
                # simg = itkImgRandomTranslation(simg,XYZRange(-30,30, -30,30, -30,30, 0.5))

            if debug: imgSave(simg)
            img = dumpForNetwork(simg, train_size)

            img_list.append([img])
            label_list.append([label])
            if len(img_list) >= batch_size:
                x = np.vstack(img_list)
                y = np.vstack(label_list)

                yield x, y
                img_list = []
                label_list = []

























def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.
    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0: raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)



def dice_coef(y_true, y_pred):
    """
    calcuate Dice Similarity Coefficient (DSC)
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    tmp = K.sum(y_true_f) + K.sum(y_pred_f)
    return (2. * intersection) / (tmp + (tmp==0)+1.0e-5)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def overlap_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    tmp=K.sum(y_true_f) + K.sum(y_pred_f) - intersection 
    return intersection / (tmp + (tmp==0)+1.0e-5)

def overlap_coef_loss(y_true, y_pred):
    return -overlap_coef(y_true, y_pred)





# def random_scale_img(img, xyz_range):
#     if len(img.shape) not in [2,3]: 
#         raise RuntimeError('\nOnly for 2D or 3D images!\n')

#     # greater than 'chance' and then rescale image
#     if random.random() < xyz_range.chance: return img

#     sitkImg = sitk.GetImageFromArray(img.astype(np.float))

#     scale_x = random.uniform(xyz_range.x_min, xyz_range.x_max)
#     scale_y = random.uniform(xyz_range.y_min, xyz_range.y_max)

#     resized_x = int(sitkImg.GetSize()[0] * scale_x)
#     resized_y = int(sitkImg.GetSize()[1] * scale_y)

#     if sitkImg.GetDimension()==2: 
#         itkImg1 = sitk_resample(sitkImg, target_spacing=None, target_size=[resized_x, resized_y])
#     else:
#         scale_z = random.uniform(xyz_range.z_min, xyz_range.z_max)
#         resized_z = int(sitkImg.GetSize()[2] * scale_z)
    
#         if (sitkImg.GetSize()[0]==3) and (sitkImg.GetSize()[1]!=3) and (sitkImg.GetSize()[2]!=3):
#             resized_x = sitkImg.GetSize()[0] # R-G-B image, only resize y-z-axis

#         sitkImg1 = sitk_resample(sitkImg, target_spacing=None, target_size=[resized_x, resized_y, resized_z])

#     temp = sitk.GetArrayFromImage(sitkImg1)   
#     img = image_central_crop(temp, img.shape) # keep image shape fixed
#     return img



# ## A rigid 3D transform with rotation in radians around a fixed center with translation
# def random_rotate_translate_img(img, rotate_center=None, rotate=[-np.pi/4., np.pi/4], translate=[-5.,5.], random_chance=0.5):
#     if len(img.shape) not in [2,3]:
#         raise RuntimeError('\nOnly for 2D or 3D images!\n')
    
#     if type(translate) not in [list, tuple]:
#         translate = [translate]*2
#     else:
#         if len(translate) > 2:
#             raise RuntimeError("'translate' should be a range in [min, max]") 

#     if type(rotate) not in [list, tuple]:
#         rotate = [rotate]*2
#     else:
#         if len(rotate) > 2:
#             raise RuntimeError("'rotate' should be a range in [min, max]") 

#     # greater than 'chance' and then transform image
#     if random.random() < random_chance: return img


#     sitkImg = sitk.GetImageFromArray(img.astype(np.float))
#     rotate_center = [ (i)/2. for i in sitkImg.GetSize()]
    
#     theta_x = random.uniform(rotate[0], rotate[1]) # in radians
#     theta_y = random.uniform(rotate[0], rotate[1])
#     translate_xy = [random.uniform(translate[0],translate[1]), random.uniform(translate[0],translate[1])]   

#     if sitkImg.GetDimension()==2: 
#         TfEuler = sitk.Euler3DTransform(rotate_center, theta_x, theta_y, translate_xy)
#     else: 
#         theta_z     = random.uniform(rotate[0], rotate[1])
#         translate_z = [random.uniform(translate[0],translate[1])]

#         if img.shape[2]==3: # R-G-B image, only rotate x-axis and translate y-z-axis
#             theta_y = 0.0
#             theta_z = 0.0
#             translate_xy[0]=0

#         TfEuler = sitk.Euler3DTransform(rotate_center, theta_x, theta_y, theta_z, translate_xy+translate_z)

#     sitkImg1 = sitk.Resample(sitkImg, sitkImg, TfEuler, sitk.sitkNearestNeighbor, 0, sitk.sitkFloat32)
#     img1 = sitk.GetArrayFromImage(sitkImg1)
#     img1 = img1.astype(img.dtype)
#     return img1



# def random_flip_img(img, random_chance=0.5):
#     img0 = img
#     if len(img)==2: img = img.reshape(list(img.shape)+[1])

#     sitkImg = sitk.GetImageFromArray(img.astype(np.float))

#     if random.random()>random_chance: sitkImg = sitk.Flip(sitkImg, [True,False,False])
#     if random.random()>random_chance: sitkImg = sitk.Flip(sitkImg, [False,True,False])
#     if random.random()>random_chance: sitkImg = sitk.Flip(sitkImg, [False,False,True])

#     img = sitk.GetArrayFromImage(sitkImg)
#     if len(img0)==2: img = np.squeeze(img) # original shape

#     img = img.astype(img0.dtype)
#     return img

  

# def random_gaussian_noise_img(img, noise_level): #  "noise_level: 0-1"
#     # if random.random() < chance: return img

#     random_value = np.random.normal(0, 1.0, img.shape) # (0,1)
#     noise_coefficient = random_value*noise_level # noise_level in (-0.15, 0.15)

#     noiseImg = img*(1 + noise_coefficient)
#     noiseImg = truncate_img_intensity(noiseImg, img.dtype)
#     return noiseImg



# def random_gaussian_noise_img(img, noiseStd=1.0, noiseMean=0.0, random_chance=0.5):
#     # add gaussian noise to image

#     if noiseStd < 0: raise RuntimeError("noise standard deviation should be not less than 0.0")
    
#     # greater than 'chance' and then transform image
#     if random.random() < random_chance: return img

#     random_coefficient = random.random()
#     sitkImg = sitk.GetImageFromArray(img.astype(np.float)) #
#     temp    = sitk.AdditiveGaussianNoise(sitkImg, noiseStd*random_coefficient, noiseMean)
#     # temp = sitk.SaltAndPepperNoise(sitkImg, 0.01)
#     # temp = sitk.ShotNoise(sitkImg, 0.5)
#     # temp = sitk.SpeckleNoise(sitkImg, 0.05)

#     temp1 = sitk.GetArrayFromImage(temp)
#     temp1 = truncate_img_intensity(temp1, img.dtype)
#     temp1 = temp1.astype(img.dtype)
#     return temp1



# def imgMakeRGBA(imgList, dtype=sitk.sitkUInt8):
#     if len(imgList) < 3 or len(imgList) > 4:
#         raise Exception("imgList must contain 3 ([r,g,b]) or 4 ([r,g,b,a]) channels.")

#     inDatatype = Sitk2NpDataTypes[imgList[0].GetPixelID()]
#     inMin = np.iinfo(inDatatype).min
#     inMax = np.iinfo(inDatatype).max

#     outDatatype = Sitk2NpDataTypes[dtype]
#     outMin = np.iinfo(outDatatype).min
#     outMax = np.iinfo(outDatatype).max

#     castImgList = []
#     for img in imgList:
#         castImg = sitk.Cast(sitk.IntensityWindowing(img, inMin, inMax, outMin, outMax), dtype)
#         castImgList.append(castImg)

#     if len(imgList) == 3:
#         channelSize = list(imgList[0].GetSize())
#         alphaArray = outMax * np.ones(channelSize[::-1], dtype=outDatatype)
#         alphaChannel = sitk.GetImageFromArray(alphaArray)
#         alphaChannel.CopyInformation(imgList[0])
#         castImgList.append(alphaChannel)
#     return sitk.Compose(castImgList)



# def sitk_resample(sitkImg, target_spacing=None, target_size=None, interpolate=False, outsideValue=0):
#     # resample an image with specific spacing (e.g., [0.5,0.5]) and size (e.g., [256, 256])

#     if ((target_spacing is None) and (target_size is None)): 
#         raise RuntimeError("\n spacing and size cannot be 'None' simultaneously in 'sitk_resample!")

#     inSpacing = sitkImg.GetSpacing()
#     inSize    = sitkImg.GetSize()
#     if (target_size is None) and (target_spacing is not None):    # Set Size
#         target_size = [int(np.ceil(inSize[i]*(inSpacing[i]/target_spacing[i]))) for i in range(sitkImg.GetDimension())]
#     if (target_spacing is None) and (target_size is not None):    
#         target_spacing = [ (inSize[i]*inSpacing[i])/target_size[i] for i in range(sitkImg.GetDimension())]
    
#     # Resample input image
#     interpolator = [sitk.sitkNearestNeighbor, sitk.sitkBSpline][interpolate]
#     identityTransform = sitk.Transform()
#     identityDirection = list(sitk.AffineTransform(sitkImg.GetDimension()).GetMatrix()) 
#     return sitk.Resample(sitkImg, target_size, identityTransform, interpolator, sitkImg.GetOrigin(), target_spacing, identityDirection, outsideValue)



# def image_central_crop(img, target_size=None):
#     # crop patch with a specific size from an image

#     if len(img.shape) not in [2,3]: 
#         raise RuntimeError('\nOnly for 2D or 3D images in "image_central_crop"!\n')
    
#     if len(img.shape) != len(target_size):
#         raise RuntimeError('\nlen(img.shape) != len(target_size) in "image_central_crop"!\n\n')

#     if target_size is None: return img

#     img0 = img
#     if len(img.shape)==2: 
#         img0 = img.reshape(list(img.shape)+[1]) # 2D image to 3D image
#         target_size +=[1]                       # 2D shape to 3D shape

#     # # 先将原图像放入一个足够大的图像中
#     bigShape = [max(target_size[i], img0.shape[i]) for i in range(len(img0.shape))]
#     bigImg   = np.zeros(bigShape)

#     start = [abs(round((bigShape[i] - img0.shape[i])/2.)) for i in range(len(img0.shape))]

#     bigImg[ start[0]: start[0]+img0.shape[0], 
#             start[1]: start[1]+img0.shape[1], 
#             start[2]: start[2]+img0.shape[2] ] = img0

#     # # 然后从大的图像中取出所需大小的图像
#     cropedImg = np.zeros(target_size)
#     start = [abs(round((target_size[i] - bigImg.shape[i])/2.)) for i in range(len(img0.shape))]

#     cropedImg = bigImg[ start[0]: start[0]+target_size[0], 
#                         start[1]: start[1]+target_size[1], 
#                         start[2]: start[2]+target_size[2] ]
#     return cropedImg