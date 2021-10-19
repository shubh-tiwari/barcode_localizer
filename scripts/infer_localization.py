"""Script for inference of classifier and localization pipeline"""

import numpy as np
import cv2
import torch
from selective_search import Segmentation, compute_sim, merge_blobs, remove_duplicate

def infer_frame(frame, model):
    """Function to infer frame
    It performs selective search segmentation to get suggested objects in frame
    and performs the model predict to get score of image
    Input : 
    1. frame : Input frame image
    Output : 
    1. Frame with bounding box
    """
    
    def box_infer(bbox):
        """Infer the object present in given bounding box"""
        [q, y1, x1, y2, x2] = list(map(int,bbox*2))
        imgcrop1 = frame[y1:y2,x1:x2]
        if imgcrop1.shape[1]*imgcrop1.shape[0]<800:
            return
        if imgcrop1.shape[1]>7*imgcrop1.shape[0] or imgcrop1.shape[0]>7*imgcrop1.shape[1]:
            return
        imgcrop = cv2.resize(imgcrop1, (224,224)).transpose(2,0,1)
        #imgcrop = resizeAndPad(imgcrop1, (160,160)).transpose(2,0,1)
        imgcrop = imgcrop/255
        imgcrop = torch.from_numpy(imgcrop).float()
        imgcrop = imgcrop.unsqueeze(0)
        out = model(imgcrop.cuda())
        score = torch.exp(out).max().item()
        label = torch.exp(out).argmax().item()
        if label == 0:
            return
        return [score, y1, x1, y2, x2]
    
    seg = Segmentation(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    num_regions = seg.num_regions
    blob_array = seg.blobs
    neighbour_list = np.asarray(list(seg.get_neighbours()))
    shape = seg.mask.shape
    sim_list = np.vstack((neighbour_list.T,np.array([compute_sim(blob_array[_idx[0]],blob_array[_idx[1]],
                                                                 shape) for _idx in neighbour_list ]))).T
    while len(sim_list):
        sort_idx = np.argsort(sim_list[:,2])
        sim_list = sim_list[sort_idx]
        blob_1 = blob_array[int(sim_list[-1][0])]
        blob_2 = blob_array[int(sim_list[-1][1])]
        sim_list = sim_list[:-1]
        t = len(blob_array)
        blob_t = merge_blobs(blob_array,blob_1,blob_2,t)
        blob_array.append(blob_t)
        if len(sim_list)==0:
            break
            
        sim_list = sim_list[(sim_list[:,0]!=blob_1.blob_idx) & (sim_list[:,1]!=blob_1.blob_idx)]
        sim_list = sim_list[(sim_list[:,0]!=blob_2.blob_idx) & (sim_list[:,1]!=blob_2.blob_idx)]
        new_sim_list = np.array([[i,t,compute_sim(blob_array[i],blob_array[t],shape)] for i in blob_t.neighbours])
        if len(new_sim_list):
            sim_list = np.vstack((sim_list, new_sim_list))

        priority = []
        priority= np.arange(len(blob_array),0,-1).clip(0,(len(blob_array)+1)/2)

        bboxes = remove_duplicate(blob_array,priority)
    bboxes = np.asarray(bboxes)
    out_frame = frame.copy()
    score = np.array(list(map(box_infer, bboxes)))
    score = score[score != None]
    maxima = -1
    for elem in score:
        [q, y1, x1, y2, x2] = elem
        if q<0.6:
            continue
        if maxima<q:
            maxima = q
            box = [y1, x1, y2, x2]
    if maxima==-1:
        return
    [y1, x1, y2, x2] = box
    frame = cv2.rectangle(out_frame, (x1,y1), (x2,y2), (0,255,0), 2)
    return frame