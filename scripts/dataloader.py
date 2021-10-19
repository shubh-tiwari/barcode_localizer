import glob
import cv2
import torch
import torch.utils.data as data

from augmentation import *

class ImageDataset(data.Dataset):
    """Class to get image dataset instance from folder"""
    def __init__(self, image_path, label, augmentation=False, invert=False, 
                 mirror=True, change_channel=False, pct=1):
        """Initialize the class
        Inputs :
        1. image_path : Path containing the images
        2. label : Label to assign
        3. augmentation : If True, it calls the augment function
        4. invert : If True, image rotated by 90 degree clockwise or anticlock randomly
        5. mirror : If True, image is replaced by its mirror image
        6. change_channel : If True, R,G,B channels of image is randomly sampled
        5. pct : Percentage of image to take randomly from all image present in folder
        """
        self.imglist = glob.glob(os.path.join(image_path, '*.png'))
        if pct<1:
            self.imglist = rnd.sample(self.imglist, int(len(self.imglist)*pct))
        self.label = label
        self.augmentation = augmentation
        self.invert = invert
        self.mirror = mirror
        self.change_channel = change_channel

    def __getitem__(self, index):
        """To get item with given index"""
        label = self.label
        img = cv2.cvtColor(cv2.imread(self.imglist[index]), cv2.COLOR_BGR2RGB)
        if self.invert:
            img = rotate(img, rnd.choice([90,270]))
        if self.mirror:
            img = rotate(img, 180)
        if self.change_channel:
            img = change_colorspace(img)
        if self.augmentation:
            img = augment(img)
        img = cv2.resize(img, (224,224), cv2.INTER_CUBIC).transpose(2,0,1)
        #img = resizeAndPad(img, (160,160)).transpose(2,0,1)
        img = img/255
        img = torch.from_numpy(img).float()
        return img, label

    def __len__(self):
        """Length of the dataset"""
        return len(self.imglist)

def get_good_dataset(good, classname=1):
    """This function returns good dataset instance along with its augmented form
    """
    good_dataset = ImageDataset(good, label=classname)
    inverted_good_dataset = ImageDataset(good, label=classname, invert=True, change_channel=True)

    mirror_good_dataset = ImageDataset(good, label=classname, mirror=True, change_channel=True)
    mirror_inverted_dataset = ImageDataset(good, label=classname, mirror=True, invert=True, change_channel=True)

    aug_good_dataset = ImageDataset(good, label=classname, augmentation=True, change_channel=True, pct=1)
    aug_mirror_dataset = ImageDataset(good, label=classname, mirror=True, change_channel=True, augmentation=True, pct=1)
    aug_inverted_dataset = ImageDataset(good, label=classname, invert=True, change_channel=True, augmentation=True, pct=1)


    total_good = data.ConcatDataset([good_dataset, inverted_good_dataset, mirror_good_dataset, 
                                     mirror_inverted_dataset, aug_good_dataset, aug_mirror_dataset, 
                                 aug_inverted_dataset])
    return total_good

def get_bad_dataset(bad, classname=0):
    """This function returns good dataset instance along with its augmented form
    """
    bad_dataset = ImageDataset(bad, label=0)
    inverted_bad_dataset = ImageDataset(bad, label=0, invert=True, change_channel=True)

    aug_bad_dataset = ImageDataset(bad, label=0, augmentation=True, pct=1, change_channel=True)
    total_bad = data.ConcatDataset([bad_dataset, inverted_bad_dataset, aug_bad_dataset])
    return total_bad

def split_indices(n, val_pct, seed):
    """Randomly split the indices for training and validation dataset"""
    nval = int(n*val_pct)
    np.random.seed(seed)
    idxs = np.random.permutation(n)
    return idxs[nval:], idxs[:nval]