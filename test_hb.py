import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import (classification_report, confusion_matrix,
                             explained_variance_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from torchvision import transforms

import data_hb
from config import get_args
from haemoglobin_main import HbTrainer


def get_all_preds(model, loader,device):
    critcal_preds = []
    baseline_preds = []
    real_hb = []

    for img, img_baseline,label, hb_value,img_baseline_label,hb_class in test_loader:
        
        hb_class=one_hot(hb_class)
        img= img.to(device)
        img_baseline = img_baseline.to(device)
        label = label.to(device)
        hb_value = hb_value.to(device)
        img_baseline_label = img_baseline_label.to(device)
        hb_class = hb_class.to(device)

        y_hat = model.predict(img).detach()
       
  
        selection_probability = trainer.model(img, fw_module="selector")
        kwta_out = kwta(selection_probability)

        # selection = torch.bernoulli(kwta_out).detach()

        select_img = img * kwta_out
        # select_img = selection

        baseline_out = model(img_baseline, fw_module="baseline")
        
        baseline_pred=baseline_out.argmax(dim = 1).detach().cpu().numpy()
        pred=y_hat.argmax(dim = 1).detach().cpu().numpy()
        
        hb=hb_class.argmax(dim = 1).detach().cpu().numpy()
        

        critcal_preds.append(pred)
        baseline_preds.append(baseline_pred)
        real_hb.append(hb)

    return critcal_preds,baseline_preds,real_hb


def one_hot(label, depth=3):
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label.long().view(-1, 1))
    out.scatter_(dim=1, index=idx, value=1)
    return out

class KWTAMask(nn.Module):
    def __init__(self, n_pick=10000):
        super(KWTAMask, self).__init__()
        self.k = n_pick

    def forward(self, x):
        x_shape=x.shape
        x=x.reshape(1,-1)
        topval = x.topk(self.k, dim=1)[0][:, -1]
        topval = topval.expand(x.shape[1], x.shape[0]).permute(1, 0)
        comp = (x >= topval).to(x).reshape(x_shape)
        return comp

def load_ckpt(trainer, ckpt_path):
    model = trainer.model
    optimizer = trainer.optimizer
    state_dict = torch.load(ckpt_path,map_location='cpu')
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    epoch = state_dict["epoch"]
    trainer.epoch = epoch
    return trainer

def tensor_to_np(tensor):
    unloader = transforms.ToPILImage()

    image = tensor.cpu().clone()
    image = image.squeeze(0)
    # print(image.shape)
    image = unloader(image)
    # img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return image


def plot_confusion_matrix(cm, result_path, classes,title='Confusion Matrix'):
     
    plt.figure(figsize=(3, 4), dpi=100)
    np.set_printoptions(precision=2)
 
    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        plt.text(x_val, y_val, "%0.2f" % (c,), color="white"  if c > cm.max()/2 else "black", fontsize=10, va='center', ha='center')
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes)
    plt.yticks(xlocations, classes)
    plt.ylabel('Ground trurh')
    plt.xlabel('Predict')
    
    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', color="gray", linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.05)
 
  
    # show confusion matrix
    plt.savefig(result_path+'.png', format='png')
    # plt.show()
    
def compute_error(label,predict):
    temp=np.vstack((label,predict))


    merger=np.array(temp.T)
    # print(merger)
    data=pd.DataFrame(merger,columns=['label','predict'])
    for i in range(6,17):
        d=data[(i<data['label']) & (data['label']<i+1)]
        num=d.shape[0]
        d['error']=d['predict']-d['label']
        error=d['error'].abs().mean()
        print(i,'-',i+1,'MAE：',error,'num:',num)
        
def print_result(y_pred_Test,y_Test):
    print(y_pred_Test.shape)
    print(y_Test.shape)
    n=y_Test.shape[0]
    print("回归的R_squared值为：", r2_score(y_Test, y_pred_Test))
    print("解释方差分",explained_variance_score(y_Test, y_pred_Test))
    print("平均绝对误差",mean_absolute_error(y_Test, y_pred_Test))
    print("均方误差",mean_squared_error(y_Test, y_pred_Test))
    RR=1-((1-r2_score(y_Test,y_pred_Test))*(n-1))/(n-15-1)
    #print(RR)
    test_abs=abs(y_Test-y_pred_Test)
    for i in range(len(test_abs)):
        if test_abs[i]>2.5:
            print("下标",i)

    #print(test_abs>2.5)

    compute_error(y_Test, y_pred_Test)

       
if __name__ == '__main__':
    args = get_args()

    test_loader = data_hb.Data(args).test_loader
    trainer = HbTrainer(args)
    ckpt_path='./results/hb/seg_mobilev3_samll_075_0.59/checkpoint'
    

    trainer = load_ckpt(trainer, ckpt_path)
    model=trainer.model.to(args.device)

    re=[i for i in range(5,20)]
    rank = torch.Tensor(re).to(args.device)
    
    model.eval()
      
    i=0
    y_criticals = []
    y_baselines=[]
    y_trues = []
    
    for img_baseline,label, hb_vlaue,hb_c in test_loader:
        
        img_baseline= img_baseline.to(args.device)
        label = label.to(args.device)
        hb_vlaue = hb_vlaue.to(args.device)
        hb_c = hb_c.to(args.device)

        y_hat = model(img_baseline).detach()
        predict_hb = torch.sum(y_hat*rank, dim=1)
        predict_hb=predict_hb.detach().cpu().numpy()
        hb_vlaue=hb_vlaue.detach().cpu().numpy()
        y_criticals.append(predict_hb)
        y_trues.append(hb_vlaue)
        
    # plt.close(fig)
    y_critical = np.concatenate(y_criticals, axis=0)
    y_true = np.concatenate(y_trues, axis=0)
    print_result(y_critical,y_true)
    


        






