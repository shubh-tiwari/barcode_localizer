"""Augmentation helper functions"""
import random as rnd
import cv2
import imutils
import numpy as np

def add_noise(img):
    """Function to add random noise"""
    h,w,c = img.shape
    noise = np.random.randint(0,50,(h, w))
    zitter = np.zeros_like(img)
    zitter[:,:,1] = noise
    return cv2.add(img, zitter)

def increase_contrast(img):
    """Function to improve contrast of image radomly"""
    rnd_clip = rnd.uniform(1,3)
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab_img)
    clahe = cv2.createCLAHE(clipLimit = rnd_clip, tileGridSize=(3,3))
    return clahe.apply(l)

def rotate(img, angle):
    """Function to rotate the image at given angle"""
    return imutils.rotate_bound(img, angle)

def change_colorspace(img):
    """Function to perform random sampling of color channels and merge them"""
    r,g,b = cv2.split(img)
    return cv2.merge(rnd.sample([b,g,r],3))

def random_crop(img, height, width):
    """Function to perform random cropping from image of given size"""
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = rnd.randint(0, img.shape[1] - width)
    y = rnd.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    return img

def resizeAndPad(img, size):
    """Function to scale the image by keeping the aspect ratio of the given object same
    by adding padding on the remaining parts"""
    h, w = img.shape[:2]
    sh, sw = size
    aspect = w/h
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: #vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0
        
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, 
                                    borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
    return scaled_img

def augment(img):
    """Function to get randomly cropped image with improved contrast"""
    (h,w,c) = img.shape
    height = rnd.randint(int(h*0.8),int(h*1))
    width = rnd.randint(int(w*0.8),int(w*1))
    img = random_crop(img, height, width)
    img = cv2.cvtColor(increase_contrast(img), cv2.COLOR_GRAY2BGR)
    return img