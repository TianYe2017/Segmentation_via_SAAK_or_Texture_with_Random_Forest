# This code is the final version that conduct final output for SAAK transformed Random Forest Image Segmentation
import numpy as np
import scipy.io as sio
import scipy.ndimage
from scipy import stats
from scipy.misc import*
import os
from skimage.util import view_as_windows
from scipy.ndimage.filters import median_filter

# Load Training and Testing data
# Threshold of number of training data for each class
SAMPLE_LIMIT=10000
SAAK_STAGE=3
MIN_SAMPLE=10000

def load_data_list(img_dir, gt_dir):
    img_list = os.listdir(img_dir)
    img_list.sort()
    img_list = [os.path.join(img_dir,x) for x in img_list][:-1]

    gt_list = os.listdir(gt_dir)
    gt_list.sort()
    gt_list = [os.path.join(gt_dir,x) for x in gt_list]
    return img_list, gt_list

def Laws_Filter(debug = False):
    L5 = np.array([[1, 4, 6, 4, 1]])
    E5 = np.array([[-1, -2, 0, 2, 1]])
    S5 = np.array([[-1, 0, 2, 0, -1]])
    W5 = np.array([[-1, 2, 0, -2, 1]])
    R5 = np.array([[1, -4, 6, -4, 1]])

    Filters1d_list = [L5, E5, S5, W5, R5]
    N = len(Filters1d_list)
    Laws_filter = []

    for i in range(N):
        for j in range(N):
            filter = np.dot(Filters1d_list[i].T, Filters1d_list[j])
            Laws_filter.append(filter)
            if debug == True:
                print(filter)
    return Laws_filter

def input_generate(img_dir, gt_dir):
    img_list, gt_list = load_data_list(img_dir, gt_dir)
    print len(img_list), len(gt_list)
    N = min(len(img_list), len(gt_list))
    print N
    imgs = np.zeros((0, 321, 481, 3))
    gts = np.zeros((0, 321, 481))
    for i in range(N):
        image = imread(img_list[i]) - 128.
        if i == 100:
            print image.shape
        if image.shape[0] != 321:
            image = np.transpose(image, (1,0,2))
        image[:,:,0] = median_filter(image[:,:,0], 3)
        image[:,:,1] = median_filter(image[:,:,1], 3)
        image[:,:,2] = median_filter(image[:,:,2], 3)
        imgs = np.concatenate((imgs, image.reshape(1,image.shape[0], image.shape[1], image.shape[2])), axis = 0)
        gt_raw = sio.loadmat(gt_list[i])
        gt_sin = np.zeros((0, 321, 481))
        for j in range(gt_raw['groundTruth'].shape[1]):
            gt_si = gt_raw['groundTruth'][0,j][0,0][0]
            if gt_si.shape[0] != 321:
                gt_si = np.transpose(gt_si, (1,0))
            gt_sin = np.concatenate((gt_sin, gt_si.reshape(1,gt_si.shape[0], gt_si.shape[1])), axis = 0)
            del gt_si
        gt_sin_F = stats.mode(gt_sin, axis = 0)[0].reshape(1,321,481)
        del gt_sin
        gts = np.concatenate((gts, gt_sin_F), axis = 0)
    return imgs, gts

def img_window(imgs, gts, train = True):
    imgs_shape = imgs.shape
    no, d = imgs_shape[0], imgs_shape[3]
    if train:
        imgs = np.lib.pad(imgs, ((0, 0),(3, 3), (3, 3),(0, 0)), 'edge')
        gts = np.lib.pad(gts, ((0, 0),(3, 3), (3, 3)), 'edge')
        imgs_window = view_as_windows(imgs, (1,7,7,d), step = (1,2,2,d)).reshape(-1,7,7,d)
        gts_window = view_as_windows(gts, (1,7,7), step = (1,2,2)).reshape(-1,7,7)
    else:
        imgs = np.lib.pad(imgs, ((0, 0),(3, 3), (3, 3),(0, 0)), 'edge')
        gts = np.lib.pad(gts, ((0, 0),(3, 3), (3, 3)), 'edge')
        imgs_window = view_as_windows(imgs, (1,7,7,d), step = (1,3,3,d)).reshape(-1,7,7,d)
        gts_window = view_as_windows(gts, (1,7,7), step = (1,3,3)).reshape(-1,7,7)
    index = 0
    imgs = []
    gts = []
    for i in range(imgs_window.shape[0]):
        gt = gts_window[i]
        if np.unique(gt).shape[0] == 1:
            img = imgs_window[i]
            img = np.round(scipy.ndimage.zoom(img, (8./7, 8./7, 1), order=1))
            imgs.append(img)
            gts.append(gt[3,3])
            del img, gt
    imgs = np.array(imgs)
    gts = np.array(gts)
    del imgs_window, gts_window
    return imgs, gts

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as fp:
        pickle.dump(obj, fp)

# Path of training data and ground truth
img_dir_train = '/home/will/EE660/BSR/BSDS500/data/images/train'
gt_dir_train = '/home/will/EE660/BSR/BSDS500/data/groundTruth/train'
imgs_train, gts_train = input_generate(img_dir_train, gt_dir_train)
print imgs_train.shape, gts_train.shape

# Path of testing data and ground truth
img_dir_test = '/home/will/EE660/BSR/BSDS500/data/images/test'
gt_dir_test = '/home/will/EE660/BSR/BSDS500/data/groundTruth/test'
imgs_test, gts_test = input_generate(img_dir_test, gt_dir_test)
print imgs_test.shape, gts_test.shape

# acquire patches
imgs_window_train, gts_window_train = img_window(imgs_train, gts_train)
print (imgs_window_train.shape, gts_window_train.shape)
del imgs_train, gts_train

imgs_window_test, gts_window_test = img_window(imgs_test, gts_test, train = False)
print (imgs_window_test.shape, gts_window_test.shape)
del imgs_test, gts_test

# Choose training and testing samples

label_dic = {}
for lab in gts_window_train:
    if int(lab) not in label_dic.keys():
        label_dic[int(lab)] = 0
    label_dic[int(lab)] = label_dic[int(lab)] + 1
print label_dic
label_list = []
for key in label_dic.keys():
    if label_dic[key] > MIN_SAMPLE:
        label_list.append(key)
print label_list

input_imgs_train = []
input_gts_train= []
step = 0
for lab in label_list:
    ind = np.where( gts_window_train == lab )[0]
    ind = ind[::ind.shape[0] / SAMPLE_LIMIT]
    gt_temp = gts_window_train[ind]
    img_temp = imgs_window_train[ind]
    if step == 0:
        input_imgs_train = img_temp
        input_gts_train = gt_temp
    else:
        input_imgs_train = np.concatenate((input_imgs_train, img_temp), axis = 0)
        input_gts_train = np.concatenate((input_gts_train, gt_temp), axis = 0)
    step += 1
ind = np.random.permutation(input_gts_train.shape[0])
input_imgs_train = input_imgs_train[ind]
input_gts_train = input_gts_train[ind]
print input_imgs_train.shape, input_gts_train.shape

del imgs_window_train, gts_window_train

input_imgs_test = []
input_gts_test= []
step = 0
for lab in label_list:
    ind = np.where( gts_window_test == lab )[0]
    ind = ind[::ind.shape[0] / (SAMPLE_LIMIT / 10) + 1]
    gt_temp = imgs_window_test[ind]
    img_temp = gts_window_test[ind]
    if step == 0:
        input_imgs_test = img_temp
        input_gts_test = gt_temp
    else:
        input_imgs_test = np.concatenate((input_imgs_test, img_temp), axis = 0)
        input_gts_test = np.concatenate((input_gts_test, gt_temp), axis = 0)
    step += 1

del imgs_window_test, gts_window_test

temp = input_imgs_test
input_imgs_test = input_gts_test
input_gts_test = temp
print input_imgs_test.shape, input_gts_test.shape

# SAAK transform

import torch
import argparse
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
from sklearn.decomposition import PCA
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.externals import joblib
from matplotlib.pyplot import imshow,show,imsave

def PCA_and_augment(data_in, ver):
    # data reshape
    data=np.reshape(data_in,(data_in.shape[0],-1))
    print 'PCA_and_augment: {}'.format(data.shape)
    # mean removal
    mean = np.mean(data, axis=0)
    datas_mean_remov = data
    print 'PCA_and_augment meanremove shape: {}'.format(datas_mean_remov.shape)

    # PCA, retain all components
    pca=PCA()
    pca.fit(datas_mean_remov)
    Energy = np.cumsum(pca.explained_variance_ratio_)
    f_num = np.count_nonzero(Energy < 0.999)
    print f_num
    comps=pca.components_[:f_num,:]
    print comps.shape
    joblib.dump(pca, './PCA/pca_for_' + 'stage_' + str(ver) + '.pkl')

    # augment, DC component doesn't
    comps_aug=[vec*(-1) for vec in comps[:-1]]
    comps_complete=np.vstack((comps,comps_aug))
    print 'PCA_and_augment comps_complete shape: {}'.format(comps_complete.shape)
    return comps_complete

from itertools import product
def fit_pca_shape(datasets,depth):
    factor=np.power(2,depth)
    length=32/factor
    print 'fit_pca_shape: length: {}'.format(length)
    idx1=range(0,length,2)
    idx2=[i+2 for i in idx1]
    print 'fit_pca_shape: idx1: {}'.format(idx1)
    data_lattice=[datasets[:,:,i:j,k:l] for ((i,j),(k,l)) in product(zip(idx1,idx2),zip(idx1,idx2))]
    data_lattice=np.array(data_lattice)
    print 'fit_pca_shape: data_lattice.shape: {}'.format(data_lattice.shape)

    #shape reshape
    data=np.reshape(data_lattice,(data_lattice.shape[0]*data_lattice.shape[1],data_lattice.shape[2],2,2))
    print 'fit_pca_shape: reshape: {}'.format(data.shape)
    return data

def ret_filt_patches(aug_anchors,input_channels):
    shape=aug_anchors.shape[1]/4
    num=aug_anchors.shape[0]
    filt=np.reshape(aug_anchors,(num,shape,4))
    
    # reshape to kernels, (# output_channels,# input_channels,2,2)
    filters=np.reshape(filt,(num,shape,2,2))

    return filters

def conv_and_relu(filters,datasets,stride=2):
    # torch data change
    filters_t=torch.from_numpy(filters)
    datasets_t=torch.from_numpy(datasets)

    # Variables
    filt=Variable(filters_t).type(torch.FloatTensor).cuda()
    data=Variable(datasets_t).type(torch.FloatTensor).cuda()

    # Convolution
    output=F.conv2d(data,filt,stride=stride)

    # Relu
    relu_output=F.relu(output)

    return relu_output.cpu(),filt.cpu()

def one_stage_saak_trans(datasets=None,datasets_test=None,depth=0):


    # intial dataset, (60000,1,32,32)
    # channel change: 1->7
    print 'one_stage_saak_trans: datasets.shape {}'.format(datasets.shape)
    input_channels=datasets.shape[1]

    # change data shape, (14*60000,4)
    data_flatten=fit_pca_shape(datasets,depth)
    data_test_flatten=fit_pca_shape(datasets_test,depth)
    
    # augmented components, first round: (7,4), only augment AC components
    comps_complete=PCA_and_augment(data_flatten, depth + 1)
    print 'one_stage_saak_trans: comps_complete: {}'.format(comps_complete.shape)

    # get filter, (7,1,2,2) 
    filters=ret_filt_patches(comps_complete,input_channels)
    print 'one_stage_saak_trans: filters: {}'.format(filters.shape)
    
    batch_size = 100
    
    relu_output,filt=conv_and_relu(filters,datasets[:batch_size],stride=2)
    if datasets.shape[0] % batch_size == 0:
        for i in range(1,datasets.shape[0] // batch_size):
            relu_out,filt=conv_and_relu(filters,datasets[i * batch_size:min((i + 1) * batch_size,datasets.shape[0])],stride=2)
            relu_output = torch.cat((relu_output, relu_out), 0)
    else:
        for i in range(1,datasets.shape[0] // batch_size + 1):
            relu_out,filt=conv_and_relu(filters,datasets[i * batch_size:min((i + 1) * batch_size,datasets.shape[0])],stride=2)
            relu_output = torch.cat((relu_output, relu_out), 0)
        
    relu_test_output,filt=conv_and_relu(filters,datasets_test[:batch_size],stride=2)
    if datasets_test.shape[0] % batch_size == 0:
        for i in range(1,datasets_test.shape[0] // batch_size):
            relu_out,filt=conv_and_relu(filters,datasets_test[i * batch_size:min((i + 1) * batch_size,datasets_test.shape[0])],stride=2)
            relu_test_output = torch.cat((relu_test_output, relu_out), 0)
    else:
        for i in range(1,datasets_test.shape[0] // batch_size + 1):
            relu_out,filt=conv_and_relu(filters,datasets_test[i * batch_size:min((i + 1) * batch_size,datasets_test.shape[0])],stride=2)
            relu_test_output = torch.cat((relu_test_output, relu_out), 0)

    data=relu_output.data.numpy()
    data_test=relu_test_output.data.numpy()
    print 'one_stage_saak_trans: output: {}'.format(data.shape)
    print 'one_stage_saak_trans: output: {}'.format(data_test.shape)
    return data,data_test,filt,relu_output,relu_test_output

def one_stage_saak_trans_test(filters,datasets_test=None,depth=0):


    # intial dataset, (60000,1,32,32)
    # channel change: 1->7
    print 'one_stage_saak_trans: datasets.shape {}'.format(datasets_test.shape)
    input_channels=datasets_test.shape[1]
    
    batch_size = 100
    
    relu_test_output,filt=conv_and_relu(filters,datasets_test[:batch_size],stride=2)
    for i in range(1,datasets_test.shape[0] // batch_size + 1):
        relu_out,filt=conv_and_relu(filters,datasets_test[i * batch_size:min((i + 1) * batch_size,datasets_test.shape[0])],stride=2)
        relu_test_output = torch.cat((relu_test_output, relu_out), 0)
    
    data_test=relu_test_output.data.numpy()
    print 'one_stage_saak_trans: output: {}'.format(data_test.shape)
    return data_test,relu_test_output

def multi_stage_saak_trans_test(filters, input_imgs_test):
    outputs_test = []
    test_data = input_imgs_test
    test_data = np.transpose(test_data, axes = (0,3,1,2))
    test_dataset=test_data
    num=0
    img_len=test_data.shape[-1]
    while(img_len>=2):
        num+=1
        img_len/=2
    
    for i in range(num):
        print '{} stage of saak transform: '.format(i)
        test_data,output_test=one_stage_saak_trans_test(filters[i].data.numpy(),test_data,depth=i + 2)
        outputs_test.append(output_test)
        print ''

    return test_dataset,outputs_test

def multi_stage_saak_trans(input_imgs_train, input_imgs_test):
    filters = []
    outputs = []
    outputs_test = []
    
    data = input_imgs_train
    data = np.transpose(data, axes = (0,3,1,2))
    test_data = input_imgs_test
    test_data = np.transpose(test_data, axes = (0,3,1,2))
    dataset=data
    test_dataset=test_data
    num=0
    img_len=data.shape[-1]
    while(img_len>=2):
        num+=1
        img_len/=2


    for i in range(num):
        print '{} stage of saak transform: '.format(i)
        data,test_data,filt,output,output_test=one_stage_saak_trans(data,test_data,depth=i + 2)
        filters.append(filt)
        outputs.append(output)
        outputs_test.append(output_test)
        print ''


    return dataset,test_dataset,filters,outputs,outputs_test

input_imgs_train, input_imgs_test, filters,outputs,outputs_test=multi_stage_saak_trans(input_imgs_train, input_imgs_test)

def Unsign(train_data):
    filternum = (train_data.shape[1]-1)/2
    print filternum
    ta1 = np.concatenate((train_data[:,:filternum]-train_data[:,filternum+1:], train_data[:,filternum:filternum + 1]),axis=1)
    return ta1.reshape(ta1.shape[0],-1)

# post processing for SAAK transform output
imgs_train = np.zeros((outputs[0].data.numpy().shape[0],0))
imgs_test = np.zeros((outputs_test[0].data.numpy().shape[0],0))    
for i in range(1,len(outputs)):
    temp = outputs[i].data.numpy()
    temp = Unsign(temp)
    print temp.shape
    imgs_train = np.concatenate((imgs_train, temp), axis = 1)
    del temp
for i in range(1,len(outputs_test)):
    temp = outputs_test[i].data.numpy()
    temp = Unsign(temp)
    print temp.shape
    imgs_test = np.concatenate((imgs_test, temp), axis = 1)
    del temp
print imgs_train.shape
print imgs_test.shape

print imgs_train.shape
print imgs_test.shape
pca = PCA(n_components = 128)
pca.fit(imgs_train)
imgs_train = pca.transform(imgs_train)
imgs_test = pca.transform(imgs_test)
coeffi = pca.explained_variance_
type = 1
imgs_avg = np.mean(imgs_train, axis = 0)
if type is 1:
    imgs_train = imgs_train - imgs_avg
    imgs_test = imgs_test - imgs_avg
    maxi = np.amax(np.absolute(imgs_train), axis = 0)
    imgs_train = imgs_train / maxi * 100
    imgs_test = imgs_test / maxi * 100
else:
    imgs_train = imgs_train - imgs_avg
    imgs_test = imgs_test - imgs_avg
    imgs_train = imgs_train / np.sqrt(coeffi) * 100
    imgs_test = imgs_test / np.sqrt(coeffi) * 100

# K-means

# from sklearn.cluster import AgglomerativeClustering, KMeans
# from sklearn.metrics import silhouette_score
# clusF = 4
# clu = KMeans(n_clusters=clusF, 
#              init='k-means++', 
#              n_init=10, 
#              max_iter=1000, 
#              tol=0.0001, 
#              precompute_distances='auto', 
#              verbose=0, 
#              random_state=None, 
#              copy_x=True, 
#              n_jobs=1, 
#              algorithm='auto')
# clu.fit(imgs_train)
# print silhouette_score(imgs_train, clu.labels_,sample_size=50000)
# print 'Done'
# for i in range(clusF):
#     imgs_train_lab = clu.labels_
#     imgs_test_lab = clu.predict(imgs_test)
#     imgs_train_O = imgs_train[imgs_train_lab == i]
#     input_gts_train_O = input_gts_train[imgs_train_lab == i]
#     imgs_test_O = imgs_test[imgs_test_lab == i]
#     input_gts_test_O = input_gts_test[imgs_test_lab == i]
#     np.save('./image_train/image_train_' + str(i) + '.npy', imgs_train_O)
#     np.save('./image_test/image_test_' + str(i) + '.npy', imgs_test_O)
#     np.save('./gt_train/gt_train_' + str(i) + '.npy', input_gts_train_O)
#     np.save('./gt_test/gt_test_' + str(i) + '.npy', input_gts_test_O)
# del imgs_train_O, imgs_test_O, input_gts_train_O, input_gts_test_O


# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier


split = 3
dep = 11
rfc = RandomForestClassifier(n_estimators = 400, 
                             bootstrap = True,
                             oob_score = True,
                             criterion = 'gini',
                             max_features = 96,# 'auto'
                             max_depth = dep,
                             min_samples_split = int(10000 / split),
                             min_samples_leaf = int(3000 / split),
                             max_leaf_nodes = None,
                             n_jobs=-1
                            )
rfc.fit(imgs_train,input_gts_train)
input_gts_train0 = rfc.predict(imgs_train)
accuray_train = np.mean((input_gts_train0 - input_gts_train) ** 2)
print 'split is ' + str(split) + ' depth is ' + str(dep)
print 'train'
print(accuray_train, rfc.score(imgs_train,input_gts_train))
input_gts_test0 = rfc.predict(imgs_test)
accuray_test = np.mean((input_gts_test0 - input_gts_test) ** 2)
print 'test'
print(accuray_test, rfc.score(imgs_test,input_gts_test))
print("Done.")

# output test images
img_list, gt_list = load_data_list(img_dir_test, gt_dir_test)

for j in range(len(img_list)):
    image = imread(img_list[j])
    imsave('/home/will/Desktop/Origin/' + str(j) + '.png', image)
    transp = False
    if image.shape[0] != 321:
        transp = True
    d = image.shape[-1]
    print img_list[j]
    image[:,:,0] = median_filter(image[:,:,0], 3)
    image[:,:,1] = median_filter(image[:,:,1], 3)
    image[:,:,2] = median_filter(image[:,:,2], 3)
    image[:,:,0] = scipy.ndimage.gaussian_filter(image[:,:,0],sigma=0.7)
    image[:,:,1] = scipy.ndimage.gaussian_filter(image[:,:,1],sigma=0.7)
    image[:,:,2] = scipy.ndimage.gaussian_filter(image[:,:,2],sigma=0.7)
    image = np.lib.pad(image, ((3, 3), (3, 3),(0, 0)), 'edge')
    image_window = view_as_windows(image, (7,7,d), step = (1,1,d))
    image_window_shape = image_window.shape
    image_window = image_window.reshape(image_window_shape[0] * image_window_shape[1], image_window_shape[-3], image_window_shape[-2], image_window_shape[-1])
    image_ = []
    for i in range(image_window.shape[0]):
        image_.append(np.round(scipy.ndimage.zoom(image_window[i], (8./7, 8./7, 1), order=1)))
    image_window = np.array(image_)
    image_window,outputs_test=multi_stage_saak_trans_test(filters, image_window)
    img_test = np.zeros((outputs_test[0].data.numpy().shape[0],0)) 
    for i in range(1,len(outputs_test)):
        temp = outputs_test[i].data.numpy()
        temp = Unsign(temp)
        print temp.shape
        img_test = np.concatenate((img_test, temp), axis = 1)
        del temp
    img_test = pca.transform(img_test)
    coeffi = pca.explained_variance_
    type = 1
    if type is 1:
        img_test = img_test - imgs_avg
        img_test = img_test / maxi * 100
    else:
        img_test = img_test - imgs_avg
        img_test = img_test / np.sqrt(coeffi) * 100
    final = rfc.predict(img_test)
    if transp:
        final_im = final.reshape(481,321)
    else:
        final_im = final.reshape(321,481)
    imsave('/home/will/Desktop/Seg/' + str(j) + '.png', final_im)
