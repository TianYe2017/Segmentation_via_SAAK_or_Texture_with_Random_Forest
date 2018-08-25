# //---------------------------------------------------------------------------//
# This program is an early exploring of applying SAAK to segmentation
# //---------------------------------------------------------------------------//
import numpy as np
import scipy.io as sio
from scipy.misc import*
from scipy import stats
import scipy.ndimage
import os

SAMPLE_LIMIT=20000
SAAK_STAGE=3
MIN_SAMPLE=10000

def label_analysis_GT0():
    TRAIN_GROUNDTRUTH=[]
    list_index_train = os.listdir("./DATA/TRAIN/GROUNDTRUTH")
    print("collect train_mat...")
    train_cluster = []
    for i in range(len(list_index_train)):
        mat = sio.loadmat("./DATA/TRAIN/GROUNDTRUTH" + "/" + list_index_train[i])
        train_cluster.append(mat)
    for i in range(len(list_index_train)):
        mat = train_cluster[i]
        vote = np.zeros((len(mat['groundTruth'][0])))
        H = mat['groundTruth'][0][0][0, 0][0].shape[0]
        W = mat['groundTruth'][0][0][0, 0][0].shape[1]
        for h in range(H):
            for w in range(W):
                TRAIN_GROUNDTRUTH.append(mat['groundTruth'][0][0][0, 0][0][h, w])
        print("already finished " + str(i) + "frame")
    del(list_index_train)
    del(train_cluster)
    pdf = np.zeros((100,2))
    for i in range(100):
        pdf[i,0] = i + 1
    for i in range(len(TRAIN_GROUNDTRUTH)):
        pdf[TRAIN_GROUNDTRUTH[i]-1,1] += 1
    print(len(TRAIN_GROUNDTRUTH)-200*481*321)
    del(TRAIN_GROUNDTRUTH)
    print (pdf)
    return pdf
   
pdf = label_analysis_GT0()   

def white_list(pdf):
    white_list = pdf
    for i in range(100):
        if white_list[i,1] > MIN_SAMPLE:
            white_list[i,1]=1
        else:
            white_list[i,1]=0
    return white_list
            
white_list = white_list(pdf)
print(white_list)       

def Generate_Training_Image_and_Label():
    LABEL=[]
    IMAGE_BLOCK=[]
    #prepare a counting table
    counting_table = np.zeros((100,2))
    for i in range(100):
        counting_table[i,0]=i+1
    #find all training list
    list_image = os.listdir("./DATA/TRAIN/IMAGE")
    
    for i in range(len(list_image)):
    #for i in range(1):
        #read an image and its label       
        image_name = list_image[i]
        n = len(image_name)
        label_name = image_name[0:(n-3)]+"mat"
        print("image: "+ image_name +"  "+ "label: "+label_name)
        image = imread("./DATA/TRAIN/IMAGE/" + image_name)
        mat = sio.loadmat("./DATA/TRAIN/GROUNDTRUTH/" + label_name)
        H = mat['groundTruth'][0][0][0, 0][0].shape[0]
        W = mat['groundTruth'][0][0][0, 0][0].shape[1]
        #border_extend
        image = np.pad(image,3,'edge')      
        for y in range(H):
            for x in range(W):
                label_of_pixel = mat['groundTruth'][0][0][0, 0][0][y, x]
                if white_list[label_of_pixel-1,1] > 0.5 and counting_table[label_of_pixel-1,1]<SAMPLE_LIMIT:                    
                    LABEL.append(mat['groundTruth'][0][0][0, 0][0][y, x])
                    counting_table[label_of_pixel-1,1] = counting_table[label_of_pixel-1,1] + 1
                    sub_image = image[y:y+7,x:x+7,:]
                    #sub_image = np.resize(sub_image,(8,8,3))
                    sub_image = np.round(scipy.ndimage.zoom(sub_image, (8./7, 8./7, 1), order=1))
                    IMAGE_BLOCK.append(sub_image)
        print("already finish: "+ str(i) + " images")
    print (counting_table)
    return LABEL,IMAGE_BLOCK
             
label_train,image_block_train = Generate_Training_Image_and_Label()    

import torch
import argparse
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
from sklearn.decomposition import PCA
import torch.nn.functional as F
from torch.autograd import Variable

def PCA_and_augment(data_in, num_key_comp):
    # data reshape
    data=np.reshape(data_in,(data_in.shape[0],-1))
    print ('PCA_and_augment: {}'.format(data.shape))
    # mean removal
    mean = np.mean(data, axis=0)
    datas_mean_remov = data - mean
    print ('PCA_and_augment meanremove shape: {}'.format(datas_mean_remov.shape))

    # PCA, retain all components
    #pca=PCA(n_components = num_key_comp)
    pca=PCA()
    pca.fit(datas_mean_remov)
    
    eng=np.cumsum(pca.explained_variance_ratio_)
    f_num = np.count_nonzero(eng < 0.999)
    comps=pca.components_[:f_num,:]
    
    # augment, DC component doesn't
    comps_aug=[vec*(-1) for vec in comps[:-1]]
    comps_complete=np.vstack((comps,comps_aug))
    print ('PCA_and_augment comps_complete shape: {}'.format(comps_complete.shape))
    return comps_complete

from itertools import product
def fit_pca_shape(datasets,depth):
    factor=np.power(2,depth)
    length=8/factor
    print ('fit_pca_shape: length: {}'.format(length))
    idx1=range(0,int(length),2)
    idx2=[i+2 for i in idx1]
    print ('fit_pca_shape: idx1: {}'.format(idx1))
    data_lattice=[datasets[:,:,i:j,k:l] for ((i,j),(k,l)) in product(zip(idx1,idx2),zip(idx1,idx2))]
    data_lattice=np.array(data_lattice)
    print ('fit_pca_shape: data_lattice.shape: {}'.format(data_lattice.shape))

    #shape reshape
    data=np.reshape(data_lattice,(data_lattice.shape[0]*data_lattice.shape[1],data_lattice.shape[2],2,2))
    print ('fit_pca_shape: reshape: {}'.format(data.shape))
    return data

def ret_filt_patches(aug_anchors,input_channels):
    shape=int(aug_anchors.shape[1]/4)
    num=int(aug_anchors.shape[0])
    filt=np.reshape(aug_anchors,(num,shape,4))
    
    # reshape to kernels, (# output_channels,# input_channels,2,2)
    filters=np.reshape(filt,(num,shape,2,2))

    return filters

def conv_and_relu(filters,datasets,stride=2):
    # torch data change
    filters_t=torch.from_numpy(filters)
    datasets_t=torch.from_numpy(datasets)

    # Variables
    filt=Variable(filters_t).type(torch.FloatTensor)
    data=Variable(datasets_t).type(torch.FloatTensor)

    # Convolution
    output=F.conv2d(data,filt,stride=stride)

    # Relu
    relu_output=F.relu(output)

    return relu_output,filt

def one_stage_saak_trans(datasets=None,depth=0,num_key_comp=[5,5,5,5,5]):   
    print ('one_stage_saak_trans: datasets.shape {}'.format(datasets.shape))
    input_channels=datasets.shape[1] 
    data_flatten=fit_pca_shape(datasets,depth)    
    comps_complete=PCA_and_augment(data_flatten,num_key_comp)
    print ('one_stage_saak_trans: comps_complete: {}'.format(comps_complete.shape))   
    filters=ret_filt_patches(comps_complete,input_channels)
    print ('one_stage_saak_trans: filters: {}'.format(filters.shape))   
    relu_output,filt=conv_and_relu(filters,datasets,stride=2)
    data=relu_output.data.numpy()
    print ('one_stage_saak_trans: output: {}'.format(data.shape))
    return data,filt,relu_output,filters

def Multi_stage_saak_trans(num_stage=0,input_data=[]):
    filters = []       
    data_train = input_data   
    data_train = np.array(data_train)
    data_train = np.transpose(data_train,(0,3,1,2))
    
    saak_all_train = []
    
    num_key_comp = [3,4,7,6,8]
    for i in range(num_stage):
        print ('{} stage of saak transform_train: '.format(i))      
        data_train,filt,output,f=one_stage_saak_trans(data_train,depth=i,num_key_comp=num_key_comp[i])
        filters.append(f)
        saak_all_train.append(data_train)

#     for i in range(5):
#         print ('{} stage of saak transform_test: '.format(i))
#         relu_output,filt=conv_and_relu(filters[i],data_test,stride=2)
#         data_test=relu_output.data.numpy()   


    del data_train
    data_train=saak_all_train[0]
    print(saak_all_train[0].shape)
    print(saak_all_train[1].shape)
    print(saak_all_train[2].shape)
    
    N = saak_all_train[0].shape[0]
    data_train = data_train.reshape((N,-1))
    
    
    for i in range(1,num_stage):
        data_train = np.concatenate((data_train,saak_all_train[i].reshape((N,-1))),axis=1)
        
    print("data_train shape is:")
    print(data_train.shape)
    
    del saak_all_train
    
    return data_train, filters

saak_train,filters = Multi_stage_saak_trans(3,image_block_train)

saak_train = saak_train.reshape((saak_train.shape[0],-1))
print(saak_train.shape)

from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest

fvalue_selector = SelectKBest(f_classif, k=200)
saak_train = fvalue_selector.fit_transform(saak_train,label_train)
#saak_testm = fvalue_selector.transform(saak_testm)
print(saak_train.shape)
#print(saak_testm.shape)

train_pca = PCA()
train_pca.fit(saak_train)
eng=np.cumsum(train_pca.explained_variance_ratio_)
f_num = np.count_nonzero(eng < 0.999)
print(f_num)
saak_train=train_pca.transform(saak_train)[:,:f_num]

train_pca = PCA()
train_pca.fit(saak_train)
eng=np.cumsum(train_pca.explained_variance_ratio_)
f_num = np.count_nonzero(eng < 0.999)
print(f_num)
saak_train=train_pca.transform(saak_train)[:,:f_num]

rfc = RandomForestClassifier(n_estimators = 400, 
                             bootstrap = True,
                             oob_score = True,
                             criterion = 'gini',
                             max_features = 'auto',
                             max_depth = 10,
                             min_samples_split = 5000,
                             min_samples_leaf = 2500,
                             max_leaf_nodes = None,
                             n_jobs=-1
                            )
rfc.fit(saak_train,label_train)
print("Done.")

accuray_train = rfc.score(saak_train,label_train)
print(accuray_train)

def Generate_Test_Image_and_Label():
    LABEL=[]
    IMAGE_BLOCK=[]
    #prepare a counting table
    counting_table = np.zeros((100,2))
    for i in range(100):
        counting_table[i,0]=i+1
    #find all training list
    list_image = os.listdir("./DATA/TEST/IMAGE")
    
    for i in range(100):
    #for i in range(1):
        #read an image and its label       
        image_name = list_image[i]
        n = len(image_name)
        label_name = image_name[0:(n-3)]+"mat"
        print("image: "+ image_name +"  "+ "label: "+label_name)
        image = imread("./DATA/TEST/IMAGE/" + image_name)
        mat = sio.loadmat("./DATA/TEST/GROUNDTRUTH/" + label_name)
        H = mat['groundTruth'][0][0][0, 0][0].shape[0]
        W = mat['groundTruth'][0][0][0, 0][0].shape[1]
        #border_extend
        image = np.pad(image,3,'edge')      
        for y in range(H):
            for x in range(W):
                label_of_pixel = mat['groundTruth'][0][0][0, 0][0][y, x]
                if label_of_pixel < 100:
                    if white_list[label_of_pixel-1,1] > 0.5 and counting_table[label_of_pixel-1,1]<SAMPLE_LIMIT:                    
                        LABEL.append(mat['groundTruth'][0][0][0, 0][0][y, x])
                        counting_table[label_of_pixel-1,1] = counting_table[label_of_pixel-1,1] + 1
                        sub_image = image[y:y+7,x:x+7,:]
                        #sub_image = np.resize(sub_image,(8,8,3))
                        sub_image = np.round(scipy.ndimage.zoom(sub_image, (8./7, 8./7, 1), order=1))                       
                        IMAGE_BLOCK.append(sub_image)               
        print("already finish: "+ str(i) + " images")
    print (counting_table)
    return LABEL,IMAGE_BLOCK
        
      
label_test,image_block_test = Generate_Test_Image_and_Label()    

def SAAK_Test_Data(stage_num,dataset,filters):
    dataset=np.array(dataset)
    dataset=dataset.transpose((0,3,1,2))
    saak_all_test=[]
    for i in range(stage_num):
        print ('{} stage of saak transform_test: '.format(i))
        relu_output,filt=conv_and_relu(filters[i],dataset,stride=2)
        dataset=relu_output.data.numpy()
        saak_all_test.append(dataset)
        
    del dataset
    dataset = saak_all_test[0]
    N = saak_all_test[0].shape[0]
    
    dataset=dataset.reshape((N,-1))
    for i in range(1,stage_num):
        dataset = np.concatenate((dataset,saak_all_test[i].reshape((N,-1))),axis=1)
    
    print("test_data shape is:")
    print(dataset.shape)
    
    del saak_all_test

    return dataset

saak_test = SAAK_Test_Data(3,image_block_test,filters)
print(saak_test.shape)

saak_test = saak_test.reshape((saak_test.shape[0],-1))
print(saak_test.shape)

saak_test = saak_test.reshape((saak_test.shape[0],-1))
print(saak_test.shape)

saak_test=train_pca.transform(saak_test)[:,:f_num]
print(saak_test.shape)

accuray_test = rfc.score(saak_test,label_test)
print(accuray_test)

import scipy.misc

def read_image(index):
    list_image = os.listdir("./DATA/TEST/IMAGE")
    image = imread("./DATA/TEST/IMAGE/" + list_image[index-1])
    h = image.shape[0]
    w = image.shape[1]   
    print(h)
    print(w)
    return h,w,image   
    #imshow(image)
    #print(type(image))
    #print(image.shape)
    

H,W,image = read_image(1)

def Transfer_single_image(stage_num,image,filters):
    IMAGE_BLOCK=[]  
    saak_all_test=[]
    image = np.pad(image,3,'edge')      
    for y in range(H):
        for x in range(W):
            sub_image = image[y:y+7,x:x+7,:]
            sub_image = np.round(scipy.ndimage.zoom(sub_image, (8./7, 8./7, 1), order=1))                       
            IMAGE_BLOCK.append(sub_image)  
    data = np.array(IMAGE_BLOCK)
    del IMAGE_BLOCK
    data=data.transpose((0,3,1,2))
    
    for i in range(stage_num):
        print ('{} stage of saak transform_test: '.format(i))
        relu_output,filt=conv_and_relu(filters[i],data,stride=2)
        data=relu_output.data.numpy()
        saak_all_test.append(data)
    
    del data
    data = saak_all_test[0]
    N = data.shape[0]
    data=data.reshape((N,-1))
    
    for i in range(1,stage_num):
        data = np.concatenate((data,saak_all_test[i].reshape((N,-1))),axis=1)
    
    print("image_data shape is:")
    print(data.shape)
    
    del saak_all_test
    return data

image_data = Transfer_single_image(3,image,filters)

image_data = fvalue_selector.transform(image_data)
print(image_data.shape)
image_data=train_pca.transform(image_data)[:,:f_num]
print(saak_test.shape)

labels = rfc.predict(image_data)
print(labels.shape)

def allocate_color(index):
    R = 1.0/100.0*255.0*index
    G = 1.0/100.0*255.0*index
    B = 1.0/100.0*255.0*index
    return R,G,B

def Reconstruct_image(h,w,labels,image_original):
    image = np.zeros((h,w,3),dtype=np.uint8)
    
    for y in range(h):
        for x in range(w):
            R,G,B = allocate_color(labels[y*w + x])
            image[y][x][0] = R
            image[y][x][1] = G
            image[y][x][2] = B
    scipy.misc.imsave('processed.png', image)
    imshow(image_original)
    