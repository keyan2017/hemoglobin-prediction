from __future__ import division

import json
import math
import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from robustness.tools.helpers import AverageMeter, accuracy
from sklearn.metrics import (explained_variance_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import data_hb
from config import get_args
from EarlyStopping import EarlyStopping

pd.set_option('mode.chained_assignment', None)

low=5
high=20
#计算预测MAE
def compute_error(label,predict):
    temp=np.vstack((label,predict))
    # temp=np.concatenate((y_true,y_test),axis=0)

    merger=np.array(temp.T)
    # print(merger)
    data=pd.DataFrame(merger,columns=['label','predict'])
    # print(data)
    for i in range(6,17):
        d=data[(i<data['label']) & (data['label']<i+1)]
        num=d.shape[0]
        d['error']=d['predict']-d['label']
        error=d['error'].abs().mean()
        print(i,'-',i+1,'MAE：',error,'num:',num)
#one-hot编码
def one_hot(label, depth=3):
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label.long().view(-1, 1))
    out.scatter_(dim=1, index=idx, value=1)
    return out
#自定义卷积层
def Conv(in_channels, out_channels, kerner_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kerner_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
    )
#改进的mobilev3net模型    
class HaemoglobinModel(nn.Module):
    """
    CNN model with 2 heads and SE-block
    with multitask model learns faster
    """
    def __init__(self, encoder, encoder_channels, 
                 hb_values, output_channels=512):
        super().__init__()
        
        # encoder features (resnet50 in my case)
        # output should be bs x c x h x w
        self.encoder = encoder
        
        # sqeeze-excite
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.downsample = nn.Conv2d(encoder_channels, output_channels, 1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.nonlin1 = nn.ReLU()
        
        self.excite = nn.Conv2d(output_channels, output_channels, 1)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.nonlin2 = nn.ReLU()
        
        self.value_head = nn.Conv2d(output_channels, hb_values, 1)
        # self.classify_head = nn.Conv2d(output_channels, hb_classes, 1)

    
    def forward(self, x):
        features = self.encoder(x)
        features = self.squeeze(features)
        features = self.downsample(features)
        features = self.nonlin1(self.bn1(features))
        
        weights_logits = self.excite(features)
        features = features * weights_logits.sigmoid()
        features = self.nonlin2(self.bn2(features))
        
        # classify_logits = self.classify_head(features).view(features.size(0), -1)
        
        # max_vale,max_index=torch.max(classify_logits,1)
        
        value_logits = self.value_head(features).view(features.size(0), -1)
        
        
        return value_logits#, classify_logits
#实验对比卷积网络
class Baseline_cnn(nn.Module):
    def __init__(self):
        super(Baseline_cnn, self).__init__()
        self.conv1 = Conv(3, 16, 3, 1, 1)
        
        # self.conv2 = Conv(16, 16, 3, 1, 1)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv3 = Conv(16, 32, 3, 1, 1)
        # self.conv4 = Conv(32, 32, 3, 1, 1)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.conv5 = Conv(32, 64, 3, 1, 1)
        # self.conv6 = Conv(64, 64, 3, 1, 1)
        self.pool3 = nn.AvgPool2d(2, 2)
        self.conv7 = Conv(64, 128, 3, 1, 1)
        # self.conv8 = Conv(128, 128, 3, 1, 1)
        # self.conv9 = Conv(128, 128, 3, 1, 1)
        self.pool5 = nn.AvgPool2d(2, 2)
        self.conv10 = Conv(128, 128, 3, 1, 1)
        # self.conv11 = Conv(128, 128, 3, 1, 1)
        # self.conv12 = Conv(128, 128, 3, 1, 1)
        # self.HP = nn.Sequential(
        #     nn.MaxPool2d(2, 2),
        #     nn.AvgPool2d(kernel_size=2, stride=1)
        # )
        # self.fc1 = nn.Sequential(
        #     nn.Linear(128, 11),
        #     nn.Sigmoid()
        # )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        # x = self.conv6(x)
        x = self.pool3(x)
        x = self.conv7(x)
        # x = self.conv8(x)
        # x = self.conv9(x)
        x = self.pool5(x)
        x = self.conv10(x)
        # x = self.conv11(x)
        # x = self.conv12(x)
        
        # x = self.HP(x)
        # x = x.view((x.size(0), -1))
        # x = self.fc1(x.view((x.size(0), -1)))
        # x = F.normalize(x, p=1, dim=1)
        return x
#浓度预测模型
def HaemoglobinModelClassify():
    # num_epochs = 10
    # lr = 3e-4
    
    # resnet50_encoder = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
    # model=HaemoglobinModel(resnet50_encoder, 2048, hb_values=15, hb_classes=2,output_channels=64)
    
    # base_encoder=Baseline_cnn()
    # model=HaemoglobinModel(base_encoder,128, hb_values=15, hb_classes=2,output_channels=64)
    
    
    # mobilenet_v3 = timm.create_model('tf_mobilenetv3_large_100', pretrained=True)
    # mobilenet_v3_encoder = nn.Sequential(*list(mobilenet_v3.children())[:-4])#.to(device)
    # model = HaemoglobinModel(mobilenet_v3_encoder, 960, hb_values=15, hb_classes=2,output_channels=64)#.to(device)
    # tf_mobilenetv3_small_100(-3,1024)
    mobilenet_v3 = timm.create_model('tf_mobilenetv3_small_075', pretrained=True)
    mobilenet_v3_encoder = nn.Sequential(*list(mobilenet_v3.children())[:-3])#.to(device)
    model = HaemoglobinModel(mobilenet_v3_encoder, 1024, hb_values=15,output_channels=64)#.to(device)
    
    
    
    #mnasnet_small -3 1280
    #(tf_mobilenetv3_small_075 -3 1024)
    # optimizer = torch.optim.AdamW(model.parameters(), lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)
    return model
#模型训练类
class HbTrainer:
    
    def __init__(self, args, load_path=None):
        self.args=args
        self.device = self.args.device
        
        self.model = HaemoglobinModelClassify()
        self.model_name='seg_mobilev3_samll_075'
        
        #self.model_name='seg_resnext50_1'
        # self.model=resnext50()
        
        #self.model_name='seg_shufflenetv2'
        #self.model=shufflenetv2()
        
        #self.model_name='seg_mobilenetv2'
        #self.model=mobilenetv2()
        #self.model_name='seg_squeezenet_1'
        #self.model=squeezenet()

        #self.model_name='seg_bcnn_1'
        #self.model=BCNN(15, pretrained=True)
        
        # self.model_name='seg_resnet_cbam_1'
        # self.model=resnet_cbam.resnet18_cbam(pretrained=False)        
        
        # self.model_name='seg_wideresnet'
        # self.model=wideresnet()
        
        
        
        self.baseline_loss = F.cross_entropy
        self.critic_loss = nn.CrossEntropyLoss()
        # self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        # self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate, weight_decay=1e-3, eps=1e-8)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=6, verbose=True)

        self.epoch = 0
        self.train_history = {"CriticAcc": [], "BaselineAcc": [], "ActorLoss": []}
        self.eval_history = {}
        # time.strftime('%Y%m%d-%H%M%S')
        self.result_dir = load_path if load_path else f"./results/{args.data_type}/{self.model_name}"
        self.softmax = nn.Softmax(dim=1)
        self.criterion_select = nn.BCEWithLogitsLoss()
    
    def print_result(self):
        ckpt_path=self.result_dir
        # print(ckpt_path)
        y_pred_Test_path=os.path.join(ckpt_path,'pred-dir/y_pred-Test.npy')
        y_pred_Train_path=os.path.join(ckpt_path,'pred-dir/y_pred-Train.npy')
        y_Test_path=os.path.join(ckpt_path,'pred-dir/y-Test.npy')
        y_Train_path=os.path.join(ckpt_path,'pred-dir/y-Train.npy')

        performance_path=os.path.join(ckpt_path,'performance.json')
        y_pred_Test=np.load(y_pred_Test_path)
        y_pred_Train=np.load(y_pred_Train_path)
        y_Test=np.load(y_Test_path)
        y_Train=np.load(y_Train_path)

        print(y_pred_Test.shape)
        print(y_pred_Train.shape)
        print(y_Test.shape)
        print(y_Train.shape)
        print("回归的R_squared值为：", r2_score(y_Test, y_pred_Test))
        print("解释方差分",explained_variance_score(y_Test, y_pred_Test))
        print("平均绝对误差",mean_absolute_error(y_Test, y_pred_Test))
        print("均方误差",mean_squared_error(y_Test, y_pred_Test))
   


        print("回归的R_squared值为：", r2_score(y_Train, y_pred_Train))
        print("解释方差分",explained_variance_score(y_Train, y_pred_Train))
        print("平均绝对误差",mean_absolute_error(y_Train, y_pred_Train))
        print("均方误差",mean_squared_error(y_Train, y_pred_Train))
        compute_error(y_Test, y_pred_Test)
        print(self.model_name)
        
    def critic_baseline_loss(self,output,hb_lable,hb_value,hb_classes,rank):
        
        # re=[6,7,8,9,10,11,12]
        # rank = torch.Tensor([i for i in re]).cpu()
        
        # max_vale,max_index=torch.max(predict_class,1)
        
        # if max_index==0:
        #     weight=np.array([1,1,1,1,1,1,1,1,1,0,0,0,0,0,0])
        # else:
        #     weight=np.array([0,0,0,0,0,0,0,0,0,1,1,1,1,1,1])
            
        # weight = torch.Tensor(weight).to(self.args.device)

        hb_value_predict = torch.sum(output*rank, dim=1)
        criterion1 = nn.KLDivLoss(reduce=False)
        outputs = torch.log(output)
        loss = criterion1(outputs, hb_lable)
        loss1 = loss.sum()/loss.shape[0]
        criterion2 = nn.L1Loss(reduction='mean')
        loss2 = criterion2(hb_value_predict, hb_value)
        
        # criterion3 = nn.CrossEntropyLoss()
        # loss3 = criterion3(predict_class, hb_classes)
        total_loss = loss1 + loss2#+loss3
        return total_loss

    def train_step(self, train_loader):
        device = self.args.device
        self.model.train()

        CriticAcc = AverageMeter()
        BaselineAcc = AverageMeter()
        ActorLoss = AverageMeter()


        re=[i for i in range(low,high)]
        rank = torch.Tensor(re).to(device).float()
        
        b_loader = tqdm(train_loader)
        for j, inputs in enumerate(b_loader):
            b_loader.set_description(f"EpochProvision: Critic: {CriticAcc.avg}, Baseline: {BaselineAcc.avg}, Actor: {ActorLoss.avg}")
            img_baseline,label, hb_vlaue,hb_c = inputs
            img_baseline = img_baseline.to(device)
            label = label.to(device)
            hb_vlaue = hb_vlaue.to(device)
            hb_c = hb_c.to(device)

            # Select a random batch of samples
            self.optimizer.zero_grad()
            # Generate a batch of selections
            # selection_probability = self.model(img, fw_module="selector")
            # selection = torch.bernoulli(selection_probability).detach()

            # Predictor objective
            # critic_input = img * selection
            # critic_out = self.model(critic_input, fw_module="predictor")

            # critic_loss = self.critic_baseline_loss(critic_out, label,age,rank)
            
            # Baseline objective
            # baseline_out = self.model(img_baseline, fw_module="baseline")
            
            value_logits = self.model(img_baseline)
            baseline_loss = self.critic_baseline_loss(value_logits, label,hb_vlaue,hb_c,rank)
            

            # batch_data = torch.cat([selection.clone().detach(),
            #                         self.softmax(critic_out).clone().detach(),
            #                         self.softmax(baseline_out).clone().detach(),
            #                         label.float()], dim=1)
           
            # Actor objective
            # actor_loss = self.actor_loss(img_baseline_label.clone().detach(),
            #                              selection_probability.clone().detach(),
            #                              critic_loss.clone().detach(),
            #                              baseline_loss.clone().detach())

            # total_loss = actor_loss + critic_loss + baseline_loss
            baseline_loss.backward()
            self.optimizer.step()


            N = label.shape[0]



            # critic_acc = accuracy(critic_out, label)[0]
            # baseline_acc = accuracy(baseline_out, label)[0]

            # critic_age = torch.sum(critic_out*rank, dim=1)
            baseline_value = torch.sum(value_logits*rank, dim=1)

          
            # critic_acc=torch.sum(abs(critic_age-age))/N
           
            baseline_acc = torch.sum(abs(baseline_value-hb_vlaue))/N


            # CriticAcc.update(critic_acc.detach(), N)
            BaselineAcc.update(baseline_acc.detach(), N)

            # ActorLoss.update(actor_loss.detach().item(), N)

        summary = {"BaselineAcc": BaselineAcc.avg}

        return summary

    def plot_results(self):
        if not os.path.isdir(self.result_dir):
            os.makedirs(self.result_dir)

        import matplotlib.pyplot as plt
        fig_path = os.path.join(self.result_dir, "MAE") + ".png"
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        # critic_hist = np.array(self.train_history["CriticAcc"])
        base_hist = np.array(self.train_history["BaselineAcc"])
        # ax.plot(np.arange(critic_hist.shape[0]), critic_hist, label="Predictor Accuracy")
        ax.plot(np.arange(base_hist.shape[0]), base_hist, label="Baseline Accuracy")
        ax.legend()
        ax.set_title("MAE")
        ax.set_xlabel("Epoch")
        fig.savefig(fig_path)
        plt.close(fig)

        # fig_path = os.path.join(self.result_dir, "actor-loss") + ".png"
        # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        # actor_hist = np.array(self.train_history["ActorLoss"])
        # ax.plot(np.arange(actor_hist.shape[0]), actor_hist, label="Actor Loss")
        # ax.legend()
        # ax.set_title("Loss")
        # ax.set_xlabel("Epoch")
        # fig.savefig(fig_path)
        # plt.close(fig)

        for k, v in self.eval_history.items():
            fig_path = os.path.join(self.result_dir, "eval-"+k) + ".png"
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            v = np.array(v)
            ax.plot(np.arange(v.shape[0]), v)
            ax.set_title(k)
            ax.set_xlabel("Epoch")
            fig.savefig(fig_path)
            plt.close(fig)

        config = os.path.join(self.result_dir, "config.json")
        with open(config, 'w') as fp:
            json.dump(vars(self.args), fp)

    def eval_metrics(self, loader,mode="Train", feature_metrics=False, pred_metrics=True):
        result_dict = {}
        
        # re=[6,7,8,9,10,11,12]
        re=[i for i in range(low,high)]
        rank = torch.Tensor(re).to(self.args.device)

        # if feature_metrics:
        #     names = [mode + "-TPR-Mean", mode + "-TPR-STD", mode + "-FDR-Mean", mode + "-FDR-STD"]
        #     for name in names:
        #         result_dict[name] = AverageMeter()

        if pred_metrics:
            names = [mode + "-MAE"]
            for name in names:
                result_dict[name] = AverageMeter()
        g_hats, y_hats = [], []
        g_trues, y_trues = [], []
        with torch.no_grad():
            for img_baseline,label, hb_vlaue,hb_c in loader:
                img_baseline = img_baseline.to(self.args.device)
                label = label.to(self.args.device)
                hb_vlaue = hb_vlaue.to(self.args.device)
                hb_c = hb_c.to(self.args.device)

                y_hat = self.model(img_baseline)
                # g_hat = self.model.importance_score(x).detach().numpy()
                if pred_metrics:

                    # auc, apr, acc = prediction_performance_metric(y, y_hat)
                    predict_value = torch.sum(y_hat*rank, dim=1)
                    result_dict[mode + "-MAE"].update((torch.sum(abs(predict_value-hb_vlaue)).detach().cpu().numpy())/hb_vlaue.shape[0],hb_vlaue.shape[0])


                # if feature_metrics:
                #     importance_score = 1. * (g_hat > 0.5)
                #     # Evaluate the performance of feature importance
                #     mean_tpr, std_tpr, mean_fdr, std_fdr = feature_performance_metric(g.detach().numpy(), importance_score)
                #     result_dict[mode + "-TPR-Mean"].update(mean_tpr, y.shape[0])
                #     result_dict[mode + "-TPR-STD"].update(std_tpr, y.shape[0])
                #     result_dict[mode + "-FDR-Mean"].update(mean_fdr, y.shape[0])
                #     result_dict[mode + "-FDR-STD"].update(std_fdr, y.shape[0])
                # g_hats.append(g_hat)
                y_hats.append(predict_value.detach().cpu().numpy())
                # g_trues.append(g.detach().numpy())
                y_trues.append(hb_vlaue.detach().cpu().numpy())

        for metric, val in result_dict.items():
            result_dict[metric] = val.avg

        # g_hat = np.concatenate(g_hats, axis=0)
        y_hat = np.concatenate(y_hats, axis=0)
        # g_true = np.concatenate(g_trues, axis=0)
        y_true = np.concatenate(y_trues, axis=0)
        # return result_dict, g_hat, y_hat, g_true, y_true
        return result_dict, y_hat, y_true

    def eval_model(self, train_loader, test_loader,feature_metrics=False, save_arr=True):
        pred_dir = os.path.join(self.result_dir, "pred-dir")
        if not os.path.isdir(pred_dir):
            os.makedirs(pred_dir)
        self.model.eval()

        modes = [("Train", train_loader), ("Test", test_loader)]
        arrays = {}
        perf = {}
        for mode_type, loader in modes:
            perf_mode, y_pred, y_true = self.eval_metrics(loader, mode_type, feature_metrics)
            perf.update(perf_mode)
            if save_arr:
                arrays[mode_type] = {"y": y_true, "y_pred": y_pred}
            if mode_type=='Test':
               y_test_pred,y_test_true= y_pred, y_true

        perf_file = os.path.join(self.result_dir, "performance.json")
        with open(perf_file, 'w') as fp:
            json.dump(perf, fp)
        if save_arr:
            for mode, d in arrays.items():
                for arr_name, arr in d.items():
                    np.save(os.path.join(pred_dir, f"{arr_name}-{mode}"), arr)
                    
    
        return perf, arrays, y_test_pred,y_test_true

    def save_checkpoint(self):
        ckpt_file = os.path.join(self.result_dir, "checkpoint")
        state_dict = dict()
        state_dict["model"] = self.model.state_dict()
        state_dict["optimizer"] = self.optimizer.state_dict()
        state_dict["epoch"] = self.epoch
        torch.save(state_dict, ckpt_file)

    def train_model(self, train_loader, test_loader):
        device = self.args.device
        self.model.to(device)
        earlystop = EarlyStopping(self.model_name,patience = 20,verbose = True)
        # self.model.baseline.to(device)
        # self.model.critic.to(device)
        t_loader = tqdm(range(self.args.max_epochs))
        for i in t_loader:
            summary = self.train_step(train_loader)
            for key, val in summary.items():
                self.train_history[key].append(val)
            self.epoch += 1

            desc = list([f"Epoch: {self.epoch}"])
            for k, v in summary.items():
                desc.append(f"{k}: {v:.3f}")
            desc = " ".join(desc)
            t_loader.set_description(desc)

            if (self.epoch % self.args.eval_freq) == 0:
                performance, arrays,y_test_pred,y_test_true= self.eval_model(train_loader, test_loader,feature_metrics=False, save_arr=False)
                mae=performance['Test-MAE']
                # print(y1)
                # print(y2)
                r2=r2_score(y_test_true,y_test_pred)
                print("r2:",r2)
                compute_error(y_test_true,y_test_pred)
              
                print(json.dumps(performance))
                earlystop(mae,self.model)
            if(earlystop.early_stop or self.epoch==self.args.max_epochs):
            # if(1==2):
                print("Early stopping")
                save_name='checkpoint-'+self.model_name+'.pt'
                self.model.load_state_dict(torch.load(save_name))
                
                performance, _ ,y_test_pred,y_test_true= self.eval_model(train_loader, test_loader,device)
                print(json.dumps(performance))
                self.plot_results()
                self.save_checkpoint()
                break

        # performance, _ ,y_test_pred,y_test_true= self.eval_model(train_loader, test_loader,device)
        # print(json.dumps(performance))
        # self.plot_results()
        # self.save_checkpoint()
        return performance

   
if __name__=="__main__":
    args = get_args()
    train_loader = data_hb.Data(args).train_loader
    test_loader = data_hb.Data(args).test_loader
    print(args)
    
    trainer = HbTrainer(args)
    performance_dict = trainer.train_model(train_loader, test_loader)
    trainer.print_result()
    
