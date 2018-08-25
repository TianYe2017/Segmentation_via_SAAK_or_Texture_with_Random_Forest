//---------------------------------------------------------------------------//
This program colloect 24 classes (20000 samples for each class)
images->patch->Laws_filters->std->pca->add_color->random forest->process test data as train data->reconstruct image
//---------------------------------------------------------------------------//
import numpy as np
import scipy.io as sio
from scipy.misc import*
from scipy import stats
import scipy.ndimage
import os
from sklearn.externals import joblib

SAMPLE_LIMIT=20000
MIN_SAMPLE=100000
Gaussian_blur_option = True
Addition_feature_option = True
Gray_image_option = True
Keep_DC = False
STD = True
Scale = False
PCA_energy = 0.950
Test_image_index=1
THRESHOLD = 10000
NUM_LOOP = 1000
USE_EXSITED_MODEL=False

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

def Generate_Laws_Filter(debug = False):
    L5 = np.array([[1, 4, 6, 4, 1]])
    E5 = np.array([[-1, -2, 0, 2, 1]])
    S5 = np.array([[-1, 0, 2, 0, -1]])
    W5 = np.array([[-1, 2, 0, -2, 1]])
    R5 = np.array([[1, -4, 6, -4, 1]])

    #print (L5.shape)

    Filters1d_list = [L5, E5, S5, W5, R5]
    N = len(Filters1d_list)
    Laws_filter = []

    # filter sequence should be: LL LE LS LW LR EL EE ES EW ER ...RL RS RS RW RR
    for i in range(N):
        for j in range(N):
            filter = np.dot(Filters1d_list[i].T, Filters1d_list[j])
            Laws_filter.append(filter)
            if debug == True:
                print(filter)
    return Laws_filter

Laws_filters = Generate_Laws_Filter(debug = True)
     
def  Generate_Training_Image_and_Label(laws_filters=[],gray_flag = False,blur_flag = False,keep_dc = False,debug=True):
    LABEL=[]
    PIXEL_FEATURE=[]
    RESPONSE_CLUSTER=[]
    PIXEL_COLOR=[]
    #prepare a counting table
    counting_table = np.zeros((100,2))
    for i in range(100):
        counting_table[i,0]=i+1
    #find all training list
    list_image = os.listdir("./DATA/TRAIN/IMAGE")
    
    if gray_flag == False:
        for i in range(len(list_image)):
            print("processing " + str(i) + " image...")
            image_name = list_image[i]
            n = len(image_name)
            label_name = image_name[0:(n-3)]+"mat"
            print("image: "+ image_name +"  "+ "label: "+label_name)
            image = imread("./DATA/TRAIN/IMAGE/" + image_name)
            mat = sio.loadmat("./DATA/TRAIN/GROUNDTRUTH/" + label_name)
            H = mat['groundTruth'][0][0][0, 0][0].shape[0]
            W = mat['groundTruth'][0][0][0, 0][0].shape[1]
            channel_1=image[:,:,0]
            channel_2=image[:,:,1]
            channel_3=image[:,:,2]
            #border_extend
            #image = np.pad(image,2,'edge')
            #gaussian blur
            if blur_flag == True:
                channel_1 = scipy.ndimage.filters.gaussian_filter(channel_1,sigma=7.5, truncate=5.0)
                channel_2 = scipy.ndimage.filters.gaussian_filter(channel_2,sigma=7.5, truncate=5.0)
                channel_3 = scipy.ndimage.filters.gaussian_filter(channel_3,sigma=7.5, truncate=5.0)
            #subtract DC
            if keep_dc == False:
                mean_channel_1 = np.mean(channel_1)
                mean_channel_2 = np.mean(channel_2)
                mean_channel_3 = np.mean(channel_3)
                channel_1 = channel_1 - mean_channel_1
                channel_2 = channel_2 - mean_channel_2
                channel_3 = channel_3 - mean_channel_3
            if debug==True:
                print(image.shape)
            #Laws_Filter
            N = len(laws_filters)
            for i in range(N):
                response = scipy.ndimage.filters.convolve(channel_1,weights=laws_filters[i])
                RESPONSE_CLUSTER.append(response)
            for i in range(N):
                response = scipy.ndimage.filters.convolve(channel_2,weights=laws_filters[i])
                RESPONSE_CLUSTER.append(response)
            for i in range(N):
                response = scipy.ndimage.filters.convolve(channel_3,weights=laws_filters[i])
                RESPONSE_CLUSTER.append(response)
            #border_extend
            for i in range(3*N):
                RESPONSE_CLUSTER[i]=np.pad(RESPONSE_CLUSTER[i],2,'edge')
            #generate feature for each pixel
            for y in range(H):
                for x in range(W):
                    label_of_pixel = mat['groundTruth'][0][0][0, 0][0][y, x]
                    if white_list[label_of_pixel-1,1] > 0.5 and counting_table[label_of_pixel-1,1]<SAMPLE_LIMIT:
                        LABEL.append(label_of_pixel)
                        counting_table[label_of_pixel-1,1] = counting_table[label_of_pixel-1,1] + 1
                        feature = np.zeros((3*N))
                        for p in range(3*N):
                            sub_image=RESPONSE_CLUSTER[p][y:y+5,x:x+5]
                            feature[p]=np.sum(sub_image*sub_image)/25
                        PIXEL_FEATURE.append(feature)
                        color_of_pixel=np.zeros((3))
                        color_of_pixel[0]=image[y,x,0]
                        color_of_pixel[1]=image[y,x,1]
                        color_of_pixel[2]=image[y,x,2]
                        PIXEL_COLOR.append(color_of_pixel)                        
    else:
        for i in range(len(list_image)):
            print("processing " + str(i) + " image...")
            image_name = list_image[i]
            n = len(image_name)
            label_name = image_name[0:(n-3)]+"mat"
            print("image: "+ image_name +"  "+ "label: "+label_name)
            image = imread("./DATA/TRAIN/IMAGE/" + image_name)
            mat = sio.loadmat("./DATA/TRAIN/GROUNDTRUTH/" + label_name)
            H = mat['groundTruth'][0][0][0, 0][0].shape[0]
            W = mat['groundTruth'][0][0][0, 0][0].shape[1]
            channel_1=image[:,:,0]
            channel_2=image[:,:,1]
            channel_3=image[:,:,2]
            #convert to gray
            channel = channel_1*0.299+channel_2*0.587+channel_1*0.114
            #gaussian blur
            if blur_flag == True:
                channel = scipy.ndimage.filters.gaussian_filter(channel,sigma=7.5, truncate=5.0)            
            #subtract DC
            if keep_dc == False:
                mean_channel = np.mean(channel)       
                channel = channel - mean_channel      
            if debug==True:
                print(image.shape)
            #Laws_Filter
            N = len(laws_filters)
            for i in range(N):
                response = scipy.ndimage.filters.convolve(channel,weights=laws_filters[i])
                RESPONSE_CLUSTER.append(response)
            #border_extend
            for i in range(N):
                RESPONSE_CLUSTER[i]=np.pad(RESPONSE_CLUSTER[i],2,'edge')
            #generate feature for each pixel
            for y in range(H):
                for x in range(W):
                    label_of_pixel = mat['groundTruth'][0][0][0, 0][0][y, x]
                    if white_list[label_of_pixel-1,1] > 0.5 and counting_table[label_of_pixel-1,1]<SAMPLE_LIMIT:
                        LABEL.append(label_of_pixel)
                        counting_table[label_of_pixel-1,1] = counting_table[label_of_pixel-1,1] + 1
                        feature = np.zeros((N))
                        for p in range(N):
                            sub_image=RESPONSE_CLUSTER[p][y:y+5,x:x+5]
                            feature[p]=np.sum(sub_image*sub_image)/25
                        PIXEL_FEATURE.append(feature)
                        color_of_pixel=np.zeros((3))
                        color_of_pixel[0]=image[y,x,0]
                        color_of_pixel[1]=image[y,x,1]
                        color_of_pixel[2]=image[y,x,2]
                        PIXEL_COLOR.append(color_of_pixel)
    
    del RESPONSE_CLUSTER
    print(len(PIXEL_FEATURE))
    print(PIXEL_FEATURE[0].shape)
    print(len(LABEL))
    print(LABEL[0].shape)
    return PIXEL_FEATURE,PIXEL_COLOR,LABEL
        
raw_train_data, pixel_color_train, train_label = Generate_Training_Image_and_Label(laws_filters=Laws_filters,
                                                                gray_flag = Gray_image_option,
                                                                blur_flag = Gaussian_blur_option,
                                                                keep_dc = Keep_DC,
                                                                debug=False)   

def  Generate_Testing_Image_and_Label(laws_filters=[],gray_flag = False,blur_flag = False,debug=True):
    LABEL=[]
    PIXEL_FEATURE=[]
    RESPONSE_CLUSTER=[]
    PIXEL_COLOR=[]
    #prepare a counting table
    counting_table = np.zeros((100,2))
    for i in range(100):
        counting_table[i,0]=i+1
    #find all training list
    list_image = os.listdir("./DATA/TEST/IMAGE")
    
    if gray_flag == False:
        for i in range(len(list_image)):
            print("processing " + str(i) + " image...")
            image_name = list_image[i]
            n = len(image_name)
            label_name = image_name[0:(n-3)]+"mat"
            print("image: "+ image_name +"  "+ "label: "+label_name)
            image = imread("./DATA/TEST/IMAGE/" + image_name)
            mat = sio.loadmat("./DATA/TEST/GROUNDTRUTH/" + label_name)
            H = mat['groundTruth'][0][0][0, 0][0].shape[0]
            W = mat['groundTruth'][0][0][0, 0][0].shape[1]
            channel_1=image[:,:,0]
            channel_2=image[:,:,1]
            channel_3=image[:,:,2]
            #border_extend
            #image = np.pad(image,2,'edge')
            #gaussian blur
            if blur_flag == True:
                channel_1 = scipy.ndimage.filters.gaussian_filter(channel_1,sigma=7.5, truncate=5.0)
                channel_2 = scipy.ndimage.filters.gaussian_filter(channel_2,sigma=7.5, truncate=5.0)
                channel_3 = scipy.ndimage.filters.gaussian_filter(channel_3,sigma=7.5, truncate=5.0)
            #subtract DC
            mean_channel_1 = np.mean(channel_1)
            mean_channel_2 = np.mean(channel_2)
            mean_channel_3 = np.mean(channel_3)
            channel_1 = channel_1 - mean_channel_1
            channel_2 = channel_2 - mean_channel_2
            channel_3 = channel_3 - mean_channel_3
            if debug==True:
                print(image.shape)
            #Laws_Filter
            N = len(laws_filters)
            for i in range(N):
                response = scipy.ndimage.filters.convolve(channel_1,weights=laws_filters[i])
                RESPONSE_CLUSTER.append(response)
            for i in range(N):
                response = scipy.ndimage.filters.convolve(channel_2,weights=laws_filters[i])
                RESPONSE_CLUSTER.append(response)
            for i in range(N):
                response = scipy.ndimage.filters.convolve(channel_3,weights=laws_filters[i])
                RESPONSE_CLUSTER.append(response)
            #border_extend
            for i in range(3*N):
                RESPONSE_CLUSTER[i]=np.pad(RESPONSE_CLUSTER[i],2,'edge')
            #generate feature for each pixel
            for y in range(H):
                for x in range(W):
                    label_of_pixel = mat['groundTruth'][0][0][0, 0][0][y, x]
                    if label_of_pixel < 100:
                        if white_list[label_of_pixel-1,1] > 0.5 and counting_table[label_of_pixel-1,1]<SAMPLE_LIMIT:
                            LABEL.append(label_of_pixel)
                            counting_table[label_of_pixel-1,1] = counting_table[label_of_pixel-1,1] + 1
                            feature = np.zeros((3*N))
                            for p in range(3*N):
                                sub_image=RESPONSE_CLUSTER[p][y:y+5,x:x+5]
                                feature[p]=np.sum(sub_image*sub_image)/25
                            PIXEL_FEATURE.append(feature)
                            color_of_pixel=np.zeros((3))
                            color_of_pixel[0]=image[y,x,0]
                            color_of_pixel[1]=image[y,x,1]
                            color_of_pixel[2]=image[y,x,2]
                            PIXEL_COLOR.append(color_of_pixel)                        
    else:
        for i in range(len(list_image)):
            print("processing " + str(i) + " image...")
            image_name = list_image[i]
            n = len(image_name)
            label_name = image_name[0:(n-3)]+"mat"
            print("image: "+ image_name +"  "+ "label: "+label_name)
            image = imread("./DATA/TEST/IMAGE/" + image_name)
            mat = sio.loadmat("./DATA/TEST/GROUNDTRUTH/" + label_name)
            H = mat['groundTruth'][0][0][0, 0][0].shape[0]
            W = mat['groundTruth'][0][0][0, 0][0].shape[1]
            channel_1=image[:,:,0]
            channel_2=image[:,:,1]
            channel_3=image[:,:,2]
            #convert to gray
            channel = channel_1*0.299+channel_2*0.587+channel_1*0.114
            #gaussian blur
            if blur_flag == True:
                channel = scipy.ndimage.filters.gaussian_filter(channel,sigma=7.5, truncate=5.0)            
            #subtract DC
            mean_channel = np.mean(channel)       
            channel = channel - mean_channel      
            if debug==True:
                print(image.shape)
            #Laws_Filter
            N = len(laws_filters)
            for i in range(N):
                response = scipy.ndimage.filters.convolve(channel,weights=laws_filters[i])
                RESPONSE_CLUSTER.append(response)
            #border_extend
            for i in range(N):
                RESPONSE_CLUSTER[i]=np.pad(RESPONSE_CLUSTER[i],2,'edge')
            #generate feature for each pixel
            for y in range(H):
                for x in range(W):
                    label_of_pixel = mat['groundTruth'][0][0][0, 0][0][y, x]
                    if label_of_pixel < 100:
                        if white_list[label_of_pixel-1,1] > 0.5 and counting_table[label_of_pixel-1,1]<SAMPLE_LIMIT:
                            LABEL.append(label_of_pixel)
                            counting_table[label_of_pixel-1,1] = counting_table[label_of_pixel-1,1] + 1
                            feature = np.zeros((N))
                            for p in range(N):
                                sub_image=RESPONSE_CLUSTER[p][y:y+5,x:x+5]
                                feature[p]=np.sum(sub_image*sub_image)/25
                            PIXEL_FEATURE.append(feature)
                            color_of_pixel=np.zeros((3))
                            color_of_pixel[0]=image[y,x,0]
                            color_of_pixel[1]=image[y,x,1]
                            color_of_pixel[2]=image[y,x,2]
                            PIXEL_COLOR.append(color_of_pixel)
    
    del RESPONSE_CLUSTER
    print(len(PIXEL_FEATURE))
    print(PIXEL_FEATURE[0].shape)
    print(len(LABEL))
    print(LABEL[0].shape)
    print(len(PIXEL_COLOR))
    print(PIXEL_COLOR[0].shape)
    print(PIXEL_COLOR[0])
    return PIXEL_FEATURE,PIXEL_COLOR,LABEL
        
raw_test_data, pixel_color_test, test_label = Generate_Testing_Image_and_Label(laws_filters=Laws_filters,
                                                                gray_flag = Gray_image_option,
                                                                blur_flag = Gaussian_blur_option, 
                                                                debug=False)   

raw_train_data=np.array(raw_train_data)
raw_test_data=np.array(raw_test_data)
print(raw_train_data.shape)
print(raw_test_data.shape)

pixel_color_train=np.array(pixel_color_train)
pixel_color_test=np.array(pixel_color_test)
print(pixel_color_train.shape)
print(pixel_color_test.shape)

mean=np.mean(raw_train_data,axis=0)
var=np.var(raw_train_data,axis=0)
std=np.sqrt(var)

del var
if STD == True:
    train_data=(raw_train_data-mean)/std
    test_data=(raw_test_data-mean)/std
else:
    train_data=raw_train_data
    test_data=raw_test_data

del raw_train_data
del raw_test_data

joblib.dump(mean,"mean.pkl")
joblib.dump(std,"std.pkl")



print(train_data.shape)
print(train_data[0])
print(test_data.shape)
print(test_data[0])

if Scale==True:
    N = train_data.shape[1]
    for i in range(N):
        max_train = np.max(train_data[:,i])
        min_train = np.min(train_data[:,i])
        max_test = np.max(test_data[:,i])
        min_test = np.min(test_data[:,i]) 
        train_data[:,i] = (train_data[:,i] - min_train)/ (max_train-min_train)
        test_data[:,i] = (test_data[:,i] - min_test) / (max_test-min_test)
        
    print(train_data.shape)
    print(train_data[0])
    print(test_data.shape)
    print(test_data[0])
from sklearn.decomposition import PCA
train_pca = PCA()
train_pca.fit(train_data)
eng=np.cumsum(train_pca.explained_variance_ratio_)
f_num = np.count_nonzero(eng < PCA_energy)
print(f_num)
train_data=train_pca.transform(train_data)[:,:f_num]
test_data=train_pca.transform(test_data)[:,:f_num]
print(train_data.shape)
print(train_data[0])
print(test_data.shape)
print(test_data[0])
joblib.dump(train_pca, "pca.pkl")
joblib.dump(f_num,"f_num.pkl")

def add_feature(texture,color):
    color = color/255.0
    output = np.concatenate((texture,color),axis=1)
    return output
if Addition_feature_option == True:
    train_data = add_feature(train_data,pixel_color_train)
    test_data = add_feature(test_data,pixel_color_test)
    print(train_data.shape)
    print(train_data[0])
    print(test_data.shape)
    print(test_data[0])
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 300, 
                             bootstrap = True,
                             oob_score = True,
                             criterion = 'gini',
                             max_features = None,
                             max_depth = 80,
                             min_samples_split = 5000,
                             min_samples_leaf = 2500,
                             max_leaf_nodes = None,
                             n_jobs=-1
                            )
rfc.fit(train_data,train_label)
print("Done.")

joblib.dump(rfc, "rf_model.pkl")

accuracy_train = rfc.score(train_data,train_label)
print(accuracy_train)
accuracy_test = rfc.score(test_data,test_label)
print(accuracy_test)

if USE_EXSITED_MODEL == True:
    rfc = joblib.load("rf_model.pkl")
    train_pca = joblib.load("pca.pkl")
    f_num = joblib.load("f_num.pkl")
    if STD==True:
        mean = joblib.load(mean.pkl)
        std = joblib.load(std.pkl)

def read_image(index):
    list_image = os.listdir("./DATA/TEST/IMAGE")
    image = imread("./DATA/TEST/IMAGE/" + list_image[index-1])
    print(list_image[index-1])
    h = image.shape[0]
    w = image.shape[1]   
    print(h)
    print(w)
    return h,w,image   
    #imshow(image)
    #print(type(image))
    #print(image.shape)
    
Test_image_index=199
H,W,image = read_image(Test_image_index)

def Transfer_single_image(image,PCA,f_num,laws_filters=[],gray_flag = False,blur_flag = False,debug=True,
                          addition=True,std_flag=True,scale_flag=True,keep_dc=True,mean = None,std = None):
    
    PIXEL_FEATURE=[]
    RESPONSE_CLUSTER=[]
    PIXEL_COLOR=[]
    output = None
        
    if gray_flag == False:      
        H = image.shape[0]
        W = image.shape[1]
        channel_1=image[:,:,0]
        channel_2=image[:,:,1]
        channel_3=image[:,:,2]
        if blur_flag == True:
            channel_1 = scipy.ndimage.filters.gaussian_filter(channel_1,sigma=7.5, truncate=5.0)
            channel_2 = scipy.ndimage.filters.gaussian_filter(channel_2,sigma=7.5, truncate=5.0)
            channel_3 = scipy.ndimage.filters.gaussian_filter(channel_3,sigma=7.5, truncate=5.0)
        #subtract DC
        if keep_dc == False:
            mean_channel_1 = np.mean(channel_1)
            mean_channel_2 = np.mean(channel_2)
            mean_channel_3 = np.mean(channel_3)
            channel_1 = channel_1 - mean_channel_1
            channel_2 = channel_2 - mean_channel_2
            channel_3 = channel_3 - mean_channel_3
        if debug==True:
            print(image.shape)
        #Laws_Filter
        N = len(laws_filters)
        for i in range(N):
            response = scipy.ndimage.filters.convolve(channel_1,weights=laws_filters[i])
            RESPONSE_CLUSTER.append(response)
        for i in range(N):
            response = scipy.ndimage.filters.convolve(channel_2,weights=laws_filters[i])
            RESPONSE_CLUSTER.append(response)
        for i in range(N):
            response = scipy.ndimage.filters.convolve(channel_3,weights=laws_filters[i])
            RESPONSE_CLUSTER.append(response)
        #border_extend
        for i in range(3*N):
            RESPONSE_CLUSTER[i]=np.pad(RESPONSE_CLUSTER[i],2,'edge')
        #generate feature for each pixel
        for y in range(H):
            for x in range(W):                
                feature = np.zeros((3*N))
                for p in range(3*N):
                    sub_image=RESPONSE_CLUSTER[p][y:y+5,x:x+5]
                    feature[p]=np.sum(sub_image*sub_image)/25
                PIXEL_FEATURE.append(feature)
                color_of_pixel=np.zeros((3))
                color_of_pixel[0]=image[y,x,0]
                color_of_pixel[1]=image[y,x,1]
                color_of_pixel[2]=image[y,x,2]
                PIXEL_COLOR.append(color_of_pixel)        
        texture = np.array(PIXEL_FEATURE)
        if scale_flag == True:
            N = texture.shape[1]
            for i in range(N):
                max_t= np.max(texture[:,i])
                min_t= np.min(texture[:,i])
                texture[:,i] = (texture[:,i] - min_t) / (max_t - min_t)
        if std_flag == True:
#             mean=np.mean(texture,axis=0)
#             var=np.var(texture,axis=0)
#             std=np.sqrt(var)  
            texture = (texture - mean)/std        
        texture = train_pca.transform(texture)[:,:f_num]        
        color = np.array(PIXEL_COLOR)/255.0
        output_train = np.concatenate((texture,color),axis=1)
        if addition == True:
            output = output_train
        else:
            output = texture
     
    else:            
        H = image.shape[0]
        W = image.shape[1]
        channel_1=image[:,:,0]
        channel_2=image[:,:,1]
        channel_3=image[:,:,2]
        #convert to gray
        channel = channel_1*0.299+channel_2*0.587+channel_1*0.114
        #gaussian blur
        if blur_flag == True:
            channel = scipy.ndimage.filters.gaussian_filter(channel,sigma=7.5, truncate=5.0)            
        #subtract DC
        mean_channel = np.mean(channel)       
        channel = channel - mean_channel      
        if debug==True:
            print(image.shape)
        #Laws_Filter
        N = len(laws_filters)
        for i in range(N):
            response = scipy.ndimage.filters.convolve(channel,weights=laws_filters[i])
            RESPONSE_CLUSTER.append(response)
        #border_extend
        for i in range(N):
            RESPONSE_CLUSTER[i]=np.pad(RESPONSE_CLUSTER[i],2,'edge')
        #generate feature for each pixel
        for y in range(H):
            for x in range(W):                     
                feature = np.zeros((N))
                for p in range(N):
                    sub_image=RESPONSE_CLUSTER[p][y:y+5,x:x+5]
                    feature[p]=np.sum(sub_image*sub_image)/25
                PIXEL_FEATURE.append(feature)
                color_of_pixel=np.zeros((3))
                color_of_pixel[0]=image[y,x,0]
                color_of_pixel[1]=image[y,x,1]
                color_of_pixel[2]=image[y,x,2]
                PIXEL_COLOR.append(color_of_pixel)
        texture = np.array(PIXEL_FEATURE)
        if scale_flag == True:
            N = texture.shape[1]
            for i in range(N):
                max_t= np.max(texture[:,i])
                min_t= np.min(texture[:,i])
                texture[:,i] = (texture[:,i] - min_t) / (max_t - min_t)
        if std_flag == True:
#             mean=np.mean(texture,axis=0)
#             var=np.var(texture,axis=0)
#             std=np.sqrt(var)  
            texture = (texture - mean)/std
        texture = train_pca.transform(texture)[:,:f_num]
        
        color = np.array(PIXEL_COLOR)/255.0
        output_feature = np.concatenate((texture,color),axis=1)
        if addition == True:
            output = output_feature
        else:
            output = texture
       
    
    del RESPONSE_CLUSTER
#     print(len(PIXEL_FEATURE))
#     print(PIXEL_FEATURE[0].shape)
#     print(len(LABEL))
#     print(LABEL[0].shape)
#     print(len(PIXEL_COLOR))
#     print(PIXEL_COLOR[0].shape)
#     print(PIXEL_COLOR[0])
    del PIXEL_FEATURE
    del PIXEL_COLOR       
    return output

image_feature_vector_set = Transfer_single_image(image,
                                                 train_pca,
                                                 f_num,
                                                 laws_filters=Laws_filters,
                                                 gray_flag = Gray_image_option,
                                                 blur_flag = Gaussian_blur_option, 
                                                 debug=False,
                                                 addition = Addition_feature_option,
                                                 std_flag = STD,
                                                 scale_flag = Scale,
                                                 keep_dc = Keep_DC,
                                                 mean=mean,
                                                 std=std)
labels = rfc.predict(image_feature_vector_set)
print(labels.shape)
    
type_list = np.zeros((100))
N = labels.shape[0]
for i in range (N):
    type_list[labels[i]] = type_list[labels[i]] + 1
count = 0
print(type_list)
for i in range (100):
    if type_list[i] == 0:
        count=count+1
type_num = 100 - count
print(type_num)  

def optimize_type(type_list,threshold):
    N = type_list.shape[0]
    black_list = np.zeros((100))
    count=0
    for i in range(N):
        if type_list[i]<threshold:
            black_list[i] = 1
    print(black_list)
    
    index_map = np.zeros((100))
    allocate_index=1
    for i in range (N):
        if black_list[i]!= 1:
            index_map[i]=allocate_index
            allocate_index=allocate_index+1
    count = allocate_index
    print(count)
    print(index_map)    
    return count,black_list,index_map

valid_num,black_list,index_map=optimize_type(type_list,THRESHOLD)
           
def allocate_color(index,total,black_list,index_map):
    if black_list[index] == 1:
        R=0
        G=0
        B=0
    else:
        R = 1.0/total*255.0*index
        G = 1.0/total*255.0*index
        B = 1.0/total*255.0*index
    return R,G,B

import matplotlib.pyplot as plt
import random
 

def Reconstruct_image(h,w,labels,image_original,type_num,black_list,index_map):
    image = np.zeros((h,w,3),dtype=np.uint8)
    
    for y in range(h):
        for x in range(w):
            R,G,B = allocate_color(labels[y*w + x],type_num,black_list,index_map)
            image[y][x][0] = R
            image[y][x][1] = G
            image[y][x][2] = B
    scipy.misc.imsave('raw_processed1.png', image)
    return image
#     for y in range(h):
#         for x in range(w):
#             R,G,B = allocate_color_v2(labels[y*w + x])
#             image[y][x][0] = R
#             image[y][x][1] = G
#             image[y][x][2] = B
#     scipy.misc.imsave('raw_processed2.png', image)
#     return image
#     for y in range(h):
#         for x in range(w):
#             image[y][x]=labels[y*w + x]
#     imshow(image_original)
#     imshow(image)

    
    #imshow(image_original)
    
raw_processed = Reconstruct_image(H,W,labels,image,valid_num,black_list,index_map)
print(raw_processed.shape)
        