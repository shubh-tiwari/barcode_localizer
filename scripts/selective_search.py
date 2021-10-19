import numpy as np
import skimage
from skimage.io import imread
from skimage.segmentation import felzenszwalb
from skimage.transform import rescale, resize
from skimage.filters import gaussian
from sklearn.preprocessing import normalize
from scipy.ndimage.filters import convolve
from skimage.filters import gaussian

class Blob:
    """Blob definition"""
    def __init__(self,idx,blob_size=None,bbox=None):
        self.blob_idx = idx
        if not blob_size is None:
            self.size = blob_size
        if not bbox is None:
            self.bbox = bbox
        self.neighbours = set()
        self.color_hist = []
        self.texture_hist = []

class Segmentation:
    """Class to perform segmentation"""
    def __init__(self, img, img_scale=0.5, scale=40, sigma=0.2, min_size=50):
        self.img = rescale(img, img_scale, anti_aliasing=False)
        """Initialize the parameters
        1. img : Input image
        2. img_scale : Scale parameter to resize the image
        3. scale : Scale parameter for cluster segmentation
        4. sigma : Width of Gaussian kernel used in segmentation
        5. min_size : Minimum component size in segmentation
        """
        if self.img.max()>1:
            self.img /= 255.0
        self.scale = scale
        self.sigma = sigma
        self.min_size = min_size
        self.mask = None
        self.multichannel=True
        self.color_hists = []
        self.texture_hists = []
        self.blobs = []
        
    @property
    def num_regions(self):
        """Returns the number of regions suggested by segmentation
        If the segmentation is already performed.
        Else, first segmentation is performed"""
        if self.mask is None:
            self.segment()
        return np.unique(self.mask).size
    
    @property
    def blob_array(self):
        """Returns the array containing the blob instances, 
        segmentation is performed
        Else, it will return None"""
        if self.blobs==[]:
            self.blobs = self.get_blobs()
            return self.blobs
        return self.blobs
            
    def segment(self):
        """Function to perform segmentation"""
        self.mask = felzenszwalb(self.img, self.scale, self.sigma, self.min_size, multichannel=True)
        self.color_hists = self.get_color_hist()
        self.texture_hists = self.get_texture_hist()
        self.blobs = self.get_blobs()
            
    def get_color_hist(self, nbins=25):
        """Function to get color histogram"""
        bins = np.linspace(0,1,nbins+1)
        labels = range(self.num_regions + 1)
        hist = np.histogram2d(self.mask.flatten(), self.img.flatten(), bins=[labels, bins])[0]
        hist = normalize(hist,norm='l1',axis=1)
        return hist
    
    def get_texture_hist(self, nbins=10, orientations=8):
        """Function to get texture histogram"""
        grad_filter = np.array([[-1.0, 0.0, 1.0]])
        filt_img = gaussian(self.img, sigma = 1.0, multichannel = True).astype(np.float32)
        grad_x = convolve(filt_img, grad_filter)
        grad_y = convolve(filt_img, grad_filter.T)
        theta = np.arctan2(grad_y, grad_y)
        labels = range(self.num_regions + 1)
        bins_orientation = np.linspace(-np.pi, np.pi, orientations + 1)
        bins_intensity = np.linspace(0.0, 1.0, nbins + 1)
        bins = [labels, bins_orientation, bins_intensity]
        temp = np.vstack([self.mask.flatten(), theta.flatten(), filt_img.flatten()]).T
        hist = np.histogramdd(temp, bins = bins)[0]
        hist = np.reshape(hist,(self.num_regions,orientations*nbins))
        hist = normalize(hist,norm='l1',axis=1)
        return hist
    
    def get_blobs(self):
        """Function to get blob instance array"""
        blob_sizes = np.bincount(self.mask.flatten())
        blob_array = []
        for i in range(self.num_regions):
            blob_array.append(Blob(i))
            _loc = np.argwhere(self.mask==i)
            bbox = np.empty(4)
            bbox[0] = _loc[:,0].min()
            bbox[1] = _loc[:,1].min()
            bbox[2] = _loc[:,0].max()
            bbox[3] = _loc[:,1].max()
            blob_array[i].blob_size = blob_sizes[i]
            blob_array[i].bbox = bbox
            blob_array[i].color_hist = self.color_hists[i]
            blob_array[i].texture_hist = self.texture_hists[i]
        return blob_array
    
    def get_neighbours(self):
        """Function to get neighbor blob set"""
        idx_neigh = np.where(self.mask[:,:-1]!=self.mask[:,1:])
        x_neigh = np.vstack((self.mask[:,:-1][idx_neigh],self.mask[:,1:][idx_neigh])).T
        x_neigh = np.sort(x_neigh,axis=1)
        x_neigh = set([tuple(_x) for _x in x_neigh])

        idy_neigh = np.where(self.mask[:-1,:]!=self.mask[1:,:])
        y_neigh = np.vstack((self.mask[:-1,:][idy_neigh],self.mask[1:,:][idy_neigh])).T
        y_neigh = np.sort(y_neigh,axis=1)
        y_neigh = set([tuple(_y) for _y in x_neigh])

        neighbour_set = x_neigh.union(y_neigh)
        for _loc in neighbour_set:
            self.blob_array[_loc[0]].neighbours.add(_loc[1])
            self.blob_array[_loc[1]].neighbours.add(_loc[0])
        return neighbour_set

def calc_fill(blob_1,blob_2,shape):
    """Function to calculate shape similarity"""
    BBox = [[]]*4
    BBox[0] = min(blob_1.bbox[0],blob_1.bbox[0])
    BBox[1] = min(blob_1.bbox[1],blob_1.bbox[1])
    BBox[2] = max(blob_1.bbox[2],blob_1.bbox[2])
    BBox[3] = max(blob_1.bbox[3],blob_1.bbox[3])
    BBox_size = abs(BBox[0]-BBox[2])*abs(BBox[1]-BBox[3])
    fill = (BBox_size - blob_1.blob_size - blob_2.blob_size)*1.0/(shape[0]*shape[1])
    return fill

def compute_sim(blob_1,blob_2,shape):
    """Function to compute similarity"""
    similarity = 0
    similarity += np.minimum(blob_1.color_hist,blob_2.color_hist).sum()
    similarity += np.minimum(blob_1.texture_hist,blob_2.texture_hist).sum()
    similarity += 1 - (blob_1.blob_size + blob_2.blob_size)*1.0/(shape[0]*shape[1])
    similarity += 1 - calc_fill(blob_1, blob_2, shape)
    return similarity

def merge_blobs(blob_array,blob_1,blob_2,t):
    """Function to merge to blobs and return new blob formed by their merging"""
    blob_t = Blob(t)
    blob_t.blob_size = blob_1.blob_size + blob_2.blob_size
    blob_t.neighbours = blob_1.neighbours.union(blob_2.neighbours)
    
    for idx in blob_1.neighbours:
        if idx ==t: continue
        blob_array[idx].neighbours.remove(blob_1.blob_idx)
        blob_array[idx].neighbours.add(t)

    for idx in blob_2.neighbours:
        if idx==t: continue
        blob_array[idx].neighbours.remove(blob_2.blob_idx)
        blob_array[idx].neighbours.add(t)

    blob_t.neighbours.remove(blob_1.blob_idx)
    blob_t.neighbours.remove(blob_2.blob_idx)

    blob_t.bbox = np.empty(4)
    blob_t.bbox[0] = min(blob_1.bbox[0], blob_2.bbox[0])
    blob_t.bbox[1] = min(blob_1.bbox[1], blob_2.bbox[1])
    blob_t.bbox[2] = max(blob_1.bbox[2], blob_2.bbox[2])
    blob_t.bbox[3] = max(blob_1.bbox[3], blob_2.bbox[3])
    
    # Merge color_hist
    blob_t.color_hist = (blob_1.color_hist*blob_1.blob_size + blob_2.color_hist*blob_2.blob_size)/blob_t.blob_size
    blob_t.texture_hist = (blob_1.texture_hist*blob_1.blob_size + blob_2.texture_hist*blob_2.blob_size)/blob_t.blob_size
    return blob_t

def remove_duplicate(blob_array,priority):
    """Remove duplicate blobs"""
    boxes = [ tuple(box.bbox) for box in blob_array]
    priority = np.asarray([p for p in priority])
    unq_boxes = set(boxes)
    boxes = np.asarray(boxes)
    
    sort_idx = np.argsort(priority)
    priority = priority[sort_idx]
    boxes = boxes[sort_idx]
    bboxes = []
    for box,p in zip(boxes,priority):
        if tuple(box) in unq_boxes:
            bboxes.append(np.append(p,box))
            unq_boxes.remove(tuple(box))
    return bboxes