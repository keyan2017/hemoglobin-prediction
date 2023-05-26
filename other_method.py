import csv
import itertools
import os

import cv2
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from config_other import get_args
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              RandomForestClassifier, RandomForestRegressor,
                              VotingClassifier)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix,
                             plot_roc_curve, roc_auc_score, roc_curve)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier

mpl.rcParams[u'font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] =False

from sklearn.metrics import mean_absolute_error  # 平方绝对误差
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import r2_score  # R square

class ConjFeat:
    def rgb_to_bgr(self,im):
        """
         This is a utility function to turn a RGB matrix into a BGR matrix.
        """
        om = np.zeros_like(im)
        om[:,:,2] = im[:,:,0]
        om[:,:,1] = im[:,:,1]
        om[:,:,0] = im[:,:,2]
        return om
    def __init__(self,image,hhr=50,input_channels='bgr'):
        """
        hhr is a value calculated manually in the paper. The default was 50.
        image is a bgr matrix of dimensions [n,m,3]. Atention! Not a RGB image but BGR.
        If you have a RGB matrix specify input_channels=='rgb' to re-arrange.
        """
        assert len(image.shape)==3

        self.hhr=hhr
        if input_channels=='rgb':
            self.image=self.rgb_to_bgr(image)
        else:
            self.image=image
        self.entropy=0
        self.HHR=0
        self.PVM   = np.zeros((3,1))
        self.area=0
        self.brightness = 0
        self.color = ('b','g','r')
        self.PVM_12= np.zeros((3,4))
    def calc_entropy(self,v=False):
        """
        The entropy is calculated using the green channel
        set v=True for information when it is calculated
        """
        equ = cv2.equalizeHist(self.image[:,:,1])
        hist,bins      = np.histogram(equ.flatten(),256,[0,256])
        norm_const     = hist.sum()
        dist_prob      = hist/norm_const
        for pixel in range(256):
            if dist_prob[pixel]!=0:
                self.entropy = self.entropy + np.abs(dist_prob[pixel]*np.log(dist_prob[pixel]))
        self.printv(v,"Entropy:",self.entropy)
    def calc_HHR(self,v=False):
        """
        Calculate the HHR feature
        set v=True for information when it is calculated
        """
        hsv    = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        for Tresh in range(self.hhr,254):
            tt=hsv[:,:,0].item(Tresh)
            self.HHR    = self.HHR + hsv[:,:,0].item(Tresh)/(self.image.shape[0]*self.image.shape[1])
        self.printv(v,"HHR:",self.HHR)
    def calc_PVM(self,v=False):
        """
        Calculate the PVM feature
        set v=True for information when it is calculated
        """
        Area   = np.shape(self.image)[0]*np.shape(self.image)[1];
        self.area=Area
        Ks     = round(0.4*Area)
        Ke     = round(0.6*Area)
        
        for i,col in enumerate(self.color):
            array_rgb = np.reshape(self.image[:,:,i], -1)
            array_sort= np.sort(array_rgb, axis=None)
            for value in range(Ks, Ke+1):
                self.PVM[i] = self.PVM[i] + array_sort[value]/(Ke-Ks+1)  
        self.printv(v,"PVM",self.PVM)
    def calc_bright(self,v=False):
        """
        Calculate the PVM feature
        set v=True for information when it is calculated
        """
        Brightness = np.sqrt(0.241*self.image[:,:,2] + 0.691*self.image[:,:,1] + 0.68*self.image[:,:,0])
        self.brightness = np.mean(Brightness)
        self.printv(v,"Brightness",self.brightness)
    def calc_PVMper(self,v=False):
        """
        Calculate PVM by percentile
        set v=True for information when it is calculated
        """
        Ks_l    = [round(0.022*self.area), round(0.5*self.area),round(0.88*self.area)]
        Ke_l    = [round(0.12*self.area), round(0.6*self.area), round(0.98*self.area)]
        for i,col in enumerate(self.color):
            array_rgb = np.reshape(self.image[:,:,i], -1)
            array_sort= np.sort(array_rgb, axis=None)
            for count in range(0,3):
                for value in range(Ks_l[count], Ke_l[count]+1):
                    self.PVM_12[i,count+1] = self.PVM_12[i,count+1] + array_sort[value]/(Ke_l[count]-Ks_l[count]+1)
            self.PVM_12[i,0] = np.mean(self.image[:,:,i])
        self.printv(v,"PVM-12 percentile",self.PVM_12)
    def calc_all_features(self,v=False):
        """
        Calculate all the features.
        set v=True to print output when it is calculated
        """
        self.calc_entropy(v)
        self.calc_HHR(v)
        self.calc_PVM(v)
        self.calc_bright(v)
        self.calc_PVMper(v)
    def get_features(self,v=False):
        """
        Return the features as a dictionary
        """
        self.calc_all_features(v)
        od={}
        for _ in enumerate(self.PVM.flatten()):
            od['PVM-'+self.color[_[0]]]=_[1]
        for _ in enumerate(self.PVM_12.flatten()):
            od['PVM_12_'+str(_[0])]=_[1]
        od['Brightness']=self.brightness
        od['HHR']=self.HHR
        od['Entropy']=self.entropy
        return od
    def printv(self,v,comment,value):
        if v==True:
            print(comment,value)
    def color_comment(self,img):
        r,g,b = cv2.split(img)
        color_featrue = []
        # 一阶矩
        r_mean = np.mean(r)
        g_mean = np.mean(g)
        b_mean = np.mean(b)
        # 二阶矩
        r_std = np.std(r)
        g_std = np.std(g)
        b_std = np.std(b)
        #三阶矩
        r_offset = (np.mean(np.abs((r - r_mean)**3)))**(1./3)
        g_offset = (np.mean(np.abs((g - g_mean)**3)))**(1./3)
        b_offset = (np.mean(np.abs((b - b_mean)**3)))**(1./3)
        color_featrue.extend([r_mean,g_mean,b_mean,r_std,g_std,b_std,r_offset,g_offset,b_offset])
        return color_featrue
def color_comment(img):
    r,g,b = cv2.split(img)
    color_featrue = []
    # 一阶矩
    r_mean = np.mean(r)
    g_mean = np.mean(g)
    b_mean = np.mean(b)
    # 二阶矩
    r_std = np.std(r)
    g_std = np.std(g)
    b_std = np.std(b)
    #三阶矩
    r_offset = (np.mean(np.abs((r - r_mean)**3)))**(1./3)
    g_offset = (np.mean(np.abs((g - g_mean)**3)))**(1./3)
    b_offset = (np.mean(np.abs((b - b_mean)**3)))**(1./3)
    color_featrue.extend([r_mean,g_mean,b_mean,r_std,g_std,b_std,r_offset,g_offset,b_offset])
    color_featrue=np.array(color_featrue)
    return color_featrue

def colorhist(img,flag_plot=False):
    hist0 = cv2.calcHist([img],[0],None,[8],[0,255])
    hist1 = cv2.calcHist([img],[1],None,[8],[0,255])
    hist2 = cv2.calcHist([img],[2],None,[8],[0,255])
    
    if flag_plot:
        plt.figure(figsize=(6,6))
        plt.plot(range(8),hist0,label = 'B')
        plt.plot(range(8),hist1,label = 'G')
        plt.plot(range(8),hist2,label = 'R')
        plt.legend()                 
        plt.title("BGR 直方图")
        plt.show()
    
    
    # hist0 = np.cumsum(cv2.calcHist([img],[0],None,[8],[0,255]))
    # hist1 = np.cumsum(cv2.calcHist([img],[1],None,[8],[0,255]))
    # hist2 = np.cumsum(cv2.calcHist([img],[2],None,[8],[0,255]))
    
    # if flag_plot:
    #     plt.figure(figsize=(6,6))
    #     plt.plot(range(256),hist0,label = 'B')
    #     plt.plot(range(256),hist1,label = 'G')
    #     plt.plot(range(256),hist2,label = 'R')
    #     plt.legend()                 
    #     plt.title("累积直方图")
    #     plt.show()
    
    return hist0,hist1,hist2

def dect_red(image,flag_pic=True):
    if flag_pic:
        cv2.imshow("test_orig",image)
        cv2.waitKey (0)
        cv2.destroyAllWindows()
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lower_red1=np.array([0,43,46])
    upper_red1=np.array([10,255,255])
    lower_red2=np.array([156,43,46])
    upper_red2=np.array([180,255,225])
    mask1=cv2.inRange(hsv,lower_red1,upper_red1)
    mask2=cv2.inRange(hsv,lower_red2,upper_red2)
    mask=mask1+mask2
    mask3=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask4=cv2.morphologyEx(mask3,cv2.MORPH_DILATE,np.ones((3,3),np.uint8))
    res=cv2.bitwise_and(image,image,mask=mask4)
    if flag_pic:
        cv2.imshow("result",res)
        cv2.waitKey (0)
        cv2.destroyAllWindows()
    return res

def image2Matrix(image,s_height,s_width,flag_pic=True):
    h,w,ch = image.shape
    pixel=dict({0:[],1:[],2:[]})
    len_dic=[]
    for c in range(ch):
        for idx in range(h):
            for idy in range(w):
                if (image[idx][idy][c]!=0):
                    pixel[c].append(image[idx][idy][c])
        len_dic.append(len(pixel[c]))
    
    min_value=np.min(len_dic)
    for k,v in pixel.items():
        pixel[k]=v[:min_value]
    
    df_empty = pd.DataFrame(pixel)
    data_shape=df_empty.shape
    data_value=df_empty.values
    n=int(data_shape[0]**0.5)
    num=n*n
    data_value_2=data_value[:num,:]
    matrix = np.zeros((n, n,3))
    matrix[:,:,0]= data_value_2[:,0].reshape(n,n)
    matrix[:,:,1]= data_value_2[:,1].reshape(n,n)
    matrix[:,:,2]= data_value_2[:,2].reshape(n,n)
    matrix = np.array(matrix, np.uint8)
    print(matrix.shape)
    if flag_pic:
        cv2.imshow("result",matrix)
        cv2.waitKey (0)
        cv2.destroyAllWindows()
    
    s_matrix=cv2.resize(matrix,(s_height, s_width))
    print(s_matrix.shape)
    if flag_pic:
        cv2.imshow("result",s_matrix)
        cv2.waitKey (0)
        cv2.destroyAllWindows()
    return matrix,n

def is_vaild(X,Y,point): #判断像素分布点是否超出图像范围，超出返回False
    if point[0] < 0 or point[0] >= X:
        return False
    if point[1] < 0 or point[1] >= Y:
        return False
    return True
    
def getNeighbors(X,Y,x,y,dist):  # 输入图片的一个像素点的位置，返回它的8邻域
    cn1 = (x+dist,y+dist)
    cn2 = (x+dist,y)
    cn3 = (x+dist,y-dist)
    cn4 = (x,y-dist)
    cn5 = (x-dist,y-dist)
    cn6 = (x-dist,y)
    cn7 = (x-dist,y+dist)
    cn8 = (x,y+dist)
    point = (cn1,cn2,cn3,cn4,cn5,cn6,cn7,cn8)
    Cn = []
    for i in point:
        if is_vaild(X,Y,i):
            Cn.append(i)
    return Cn
        
def corrlogram(img,dist):
    xx,yy,tt = img.shape
    cgram = np.zeros((256,256),np.uint8)
    for x in range(xx):
        for y in range(yy):
            for t in range(tt):
                color_i = img[x,y,t]   # X的某一个通道的像素值
                neighbors_i = getNeighbors(xx,yy,x,y,dist)
            for j in neighbors_i:
                j0 = j[0]
                j1 = j[1]
                color_j = img[j0,j1,t]   #X的某一个邻域像素点的某一个通道的像素值
                cgram[color_i,color_j] +=  1  #统计像素值i核像素值j的个数
    return cgram

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def read_image_label(type,args):
    do_data=[]
    do_label=[]
    
    if type=='train':
        files=csv.reader(open(args.train_label, 'r'))
        root = args.train_img
        root_baseline=args.train_img_baseline
    elif type=='val':
        files=csv.reader(open(args.val_label, 'r'))
        root = args.val_img
        root_baseline=args.val_img_baseline
    else:
        print("error+++++++++")
    
    next(files) 
  
    labels = [label for label in files]
    for i,data in enumerate(labels):
        # print(i)
        name, hb_value,hb_class= data
        name=name+'.jpg'
        hb_value=float(hb_value)
        # print(name)
        # print(os.path.join(root_baseline, name))
        img=cv2.imread(os.path.join(root_baseline, name))
        # img=contrast_demo(img,0.8,3)
        
        # cf = ConjFeat(img)
        # values = cf.get_features(v=False)
        img=dect_red(img,False)
        hist0,hist1,hist2=colorhist(img)
        hist0=hist0.squeeze()
        hist1=hist1.squeeze()
        hist2=hist2.squeeze()
        color_featrue=color_comment(img)
        temp=np.concatenate((hist0,hist1,hist2,color_featrue),axis=0)
        
        do_data.append(temp)
        do_label.append(hb_value)
        
    # data_frame=pd.DataFrame(do_data)
    # data_frame=pd.DataFrame(do_data) 
    do_data=np.array(do_data)  
    do_label=np.array(do_label)
    # do_data=data_frame.values
    return do_data,do_label

def Classification_2(model, X_train, y_train, X_test, y_test):
    random_state = np.random.RandomState(0)
    model.fit(X_train, y_train)   

    # Calculating the accuracies
    print('--- Model Accuracy ---')
    score_train, score_test = model.score(X_train, y_train)*100, model.score(X_test, y_test)*100
    print('Training : %.2f [%%] \nTest :     %.2f [%%]\n'%(score_train, score_test))
    
    # Predicting on Test-set
    y_pred = model.predict(X_test)
    
    mse=mean_squared_error(y_test,y_pred)
    mae=mean_absolute_error(y_test,y_pred)
    r=r2_score(y_test,y_pred)
    print('mse:',mse)
    print('mae:',mae)
    print('r:',r)
    return mse,mae,r,model
def plot_results(mse,mae,r,clf_names):
    barWidth = 0.125
    r1 = np.arange(len(clf_names))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    # r5 = [x + barWidth for x in r4]
    # Make the plot
    
    plt.figure(figsize=(3,4),dpi=300)
    # plt.rcParams['figure.figsize'] = (5, 6)
    # plt.plot(len(mse),mse,color='#3498db', label='mse')
    # plt.plot(len(mse),mae,color='#e74c3c',  label='mae')
    # plt.plot(len(mse),r,color='#2d7f5e', label='r')
    plt.plot(mse)
    plt.plot(mae)
    plt.plot(r)
    
    # plt.bar(r1, mse, color='#3498db', width=barWidth, edgecolor='white', label='mse')
    # plt.bar(r2, mae,  color='#e74c3c', width=barWidth, edgecolor='white', label='mae')
    # plt.bar(r3, r,  color='#2d7f5e', width=barWidth, edgecolor='white', label='r')

    # plt.bar(r5, auc,  color='orange', width=barWidth, edgecolor='white', label='AUC')
 
    # Add xticks on the middle of the group bars
    plt.title('Performance comparison', fontsize = 16)
    plt.xlabel('regression', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(clf_names))], [r for r in clf_names], fontsize=11, weight='bold',rotation=85)
    plt.yticks(fontsize=13)
    min_value = np.concatenate( (mse, mae, r) ).min()
    # plt.ylim([0.95*min_value, 1.0]) 
    plt.rcParams['figure.figsize'] = (18, 4)
    plt.legend(['mse', 'mae', 'r'], fontsize=12, loc='best')
    
    # save_name='./pic/result-'+datatype+'.png'
    # plt.savefig(save_name, dpi=300)
    plt.show()
#对比实验   
def do_experiment_2(X_train, y_train, X_test, y_test):
    from sklearn import ensemble, linear_model, neighbors, svm, tree
    from sklearn.ensemble import BaggingRegressor
    from sklearn.tree import ExtraTreeRegressor  
    Clf_dict = dict()               # Create a dicitionary of several classifiers 
    Clf_dict[0] = tree.DecisionTreeRegressor()
    Clf_dict[1] = linear_model.LinearRegression()
    Clf_dict[2] = svm.SVR()
    Clf_dict[3] = neighbors.KNeighborsRegressor()
    Clf_dict[4] = ensemble.RandomForestRegressor(n_estimators=20)
    Clf_dict[5] = ensemble.AdaBoostRegressor(n_estimators=50)
    Clf_dict[6] = ensemble.GradientBoostingRegressor(n_estimators=100)
    Clf_dict[7] = ExtraTreeRegressor()
    Clf_dict[8] = BaggingRegressor()
    
    
    mse,mae,r,model_trained=dict(),dict(),dict(),dict()

    clf_names = ['dtr', 'lr', 'svr', 'kr', 'rr',
                      'adar', 'gbr','etr', 'br']
    
    for i in range(len(Clf_dict)):
        print('classify:',i)
        model= Clf_dict[i]
        mse[i], mae[i], r[i], model_trained[i]=Classification_2(model, X_train, y_train, X_test, y_test)
    
    mse=list(mse.values())
    mae=list(mae.values())
    r=list(r.values())
    plot_results(mse,mae,r,clf_names)
    
    print('mse:',mse)
    print('mae:',mae)
    print('r:',r)
    return model_trained,clf_names       

def contrast_demo(img1, c, b):  # 亮度就是每个像素所有通道都加上b
    rows, cols, chunnel = img1.shape
    blank = np.zeros([rows, cols, chunnel], img1.dtype)  # np.zeros(img1.shape, dtype=uint8)
    dst = cv2.addWeighted(img1, c, blank, 1-c, b)
    # cv2.imshow("con_bri_demo", dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return dst


if __name__ == '__main__':
    
    # 单图测试
    name='419315.jpg'
    image=cv2.imread(name)
    cv2.imshow("orgi", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    image=contrast_demo(image,0.8,3)
    
    res=dect_red(image,False)
    res=image
    hist0,hist1,hist2=colorhist(image)
    hist0=hist0.squeeze()
    hist1=hist1.squeeze()
    hist2=hist2.squeeze()
    color_featrue=color_comment(image)
    temp=np.concatenate((hist0,hist1,hist2,color_featrue),axis=0)
    print(temp.shape)
    
    print(color_featrue)
    plt.plot(color_featrue)
    plt.show()
    crgam=corrlogram(res,4)
    plt.imshow(crgam)
    plt.show()
    image_matrix,n=image2Matrix(res,64,64,True)
    print(image_matrix)
    print(n)
    
    # args = get_args()
    # train_data,train_label=read_image_label('train',args)
    # test_data,test_label=read_image_label('val',args)
    
    # train_data=normalization(train_data)
    # test_data=normalization(test_data)
    # print(test_data.shape)
    # print(test_label.shape)
    # print('ss')
    # do_experiment_2(train_data,train_label,test_data,test_label)
