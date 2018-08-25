//---------------------------------------------------------------------------//
This program use single image segmentation
image->patches->Laws_filters->std->pca->add_color->k_means->reconstruct image
//---------------------------------------------------------------------------//
import numpy as np
import scipy.io as sio
from scipy.misc import*
from scipy import stats
import scipy.ndimage
import os
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

IMAGE_INDEX = 2
pca_energy = 0.950
Gray_option = True
Blur_option = True
STD = False
Scale = True
Keep_dc = False
Addition_feature = True
cluster_num = 5

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

def read_image(index):
    list_image = os.listdir("./DATA/TEST/IMAGE")
    image = imread("./DATA/TEST/IMAGE/" + list_image[index-1])
    print(list_image[index-1])
    h = image.shape[0]
    w = image.shape[1]   
    print(h)
    print(w)
    print(list_image[index-1])
    return h,w,image,list_image[index-1]   
    #imshow(image)
    #print(type(image))
    #print(image.shape)
    
#H,W,image,name = read_image(IMAGE_INDEX)

def Transfer_single_image(image,pca_energy,laws_filters=[],gray_flag = False,blur_flag = False,
                          addition=True,std_flag=True,scale_flag=True,keep_dc=True):
   
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
            channel_1 = scipy.ndimage.filters.gaussian_filter(channel_1,sigma=3.0, truncate=5.0)
            channel_2 = scipy.ndimage.filters.gaussian_filter(channel_2,sigma=3.0, truncate=5.0)
            channel_3 = scipy.ndimage.filters.gaussian_filter(channel_3,sigma=3.0, truncate=5.0)
        #subtract DC
        if keep_dc == False:
            mean_channel_1 = np.mean(channel_1)
            mean_channel_2 = np.mean(channel_2)
            mean_channel_3 = np.mean(channel_3)
            channel_1 = channel_1 - mean_channel_1
            channel_2 = channel_2 - mean_channel_2
            channel_3 = channel_3 - mean_channel_3
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
            mean=np.mean(texture,axis=0)
            var=np.var(texture,axis=0)
            std=np.sqrt(var)  
            texture = (texture - mean)/std  
       
        train_pca = PCA()
        train_pca.fit(texture)
        eng=np.cumsum(train_pca.explained_variance_ratio_)
        f_num = np.count_nonzero(eng < pca_energy)
        print(f_num)
        texture=train_pca.transform(texture)[:,:f_num]       
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
            channel = scipy.ndimage.filters.gaussian_filter(channel,sigma=3.0, truncate=5.0)            
        #subtract DC
        if keep_dc == False:
            mean_channel = np.mean(channel)       
            channel = channel - mean_channel      
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
        
        train_pca = PCA()
        train_pca.fit(texture)
        eng=np.cumsum(train_pca.explained_variance_ratio_)
        f_num = np.count_nonzero(eng < pca_energy)
        print(f_num)
        texture=train_pca.transform(texture)[:,:f_num]     
        
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

# image_feature_vector_set = Transfer_single_image(image,
#                                                  pca_energy,
#                                                  laws_filters=Laws_filters,
#                                                  gray_flag = Gray_option,
#                                                  blur_flag = Blur_option,                                             
#                                                  addition = Addition_feature,
#                                                  std_flag = STD,
#                                                  scale_flag = Scale,
#                                                  keep_dc = Keep_dc,                                                
#                                                 )

def allocate_color(index,cluster_num):
    R = 1.0/cluster_num*255.0*index
    G = 1.0/cluster_num*255.0*index
    B = 1.0/cluster_num*255.0*index
    return R,G,B

def reconstruct_image(labels,cluster_num,H,W,image_name):
    image = np.zeros((H,W,3),dtype=np.uint8)
    for y in range (H):
        for x in range (W):
            R,G,B = allocate_color(labels[y*W + x],cluster_num)
            image[y][x][0] = R
            image[y][x][1] = G
            image[y][x][2] = B
    path = "./single_k_mean/" + image_name
    scipy.misc.imsave(path, image)
    return image
            
#new_image = reconstruct_image(labels,cluster_num,H,W,name)     
print("done")
    
def find_all_result(pca_energy,laws_filters,gray_flag,blur_flag,addition,std_flag,scale_flag,keep_dc,cluster_num):
    for i in  range(1,200):
        print("processing: " + str(i+1) + " th image")
        H,W,image,name = read_image(i)
        image_feature_vector_set = Transfer_single_image(image,
                                                 pca_energy,
                                                 laws_filters=laws_filters,
                                                 gray_flag = gray_flag,
                                                 blur_flag = blur_flag,                                             
                                                 addition = addition,
                                                 std_flag = std_flag,
                                                 scale_flag = scale_flag,
                                                 keep_dc = keep_dc
                                                        )                                                
        kmeans1 = KMeans(n_clusters=cluster_num,
                            max_iter = 1000,
                            tol = 0.00001,
                            n_jobs = -1,
                            algorithm = "full", #"elkan"
                            random_state=0).fit(image_feature_vector_set)
        labels = kmeans1.predict(image_feature_vector_set)            
        new_image = reconstruct_image(labels,cluster_num,H,W,name) 

            
            
find_all_result(pca_energy,Laws_filters,Gray_option,Blur_option,Addition_feature,STD,Scale,Keep_dc,cluster_num)           
            
