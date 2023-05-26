import csv
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import dataloader, dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader

low=5
high=20
class Data:
    def __init__(self, args):
        self.args = args
        transform_list = [
            # transforms.RandomChoice(
            #     [
            #      transforms.RandomHorizontalFlip(),
            #      transforms.RandomRotation(20),
            #      ]
            # ),
            transforms.Resize((args.height, args.width), interpolation=3),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.87, 0.79, 0.79], std=[0.04, 0.06, 0.06]) #conjunctiva_train
        ]
        transform = transforms.Compose(transform_list)

        self.train_dataset = Dataset(args, transform,'train')
        self.test_dataset = Dataset(args, transform,'val')
        
        self.train_loader = dataloader.DataLoader(self.train_dataset,
                                                  shuffle=True,
                                                  batch_size=args.train_batch_size,
                                                  num_workers=args.nThread,
                                                  drop_last=True
                                                  )
        self.test_loader = dataloader.DataLoader(self.test_dataset,
                                                  shuffle=False,
                                                  batch_size=args.val_batch_size,
                                                  num_workers=args.nThread,
                                                  drop_last=True
                                                  )


class Dataset(dataset.Dataset):
    def __init__(self, args, transform,type):
        
        self.transform = transform
        if type=='train':
            files=csv.reader(open(args.train_label, 'r'))
            # self.root = args.train_img
            self.root_baseline=args.train_img_baseline
            # self.root_baseline_label=args.train_img_baseline_label
            
        elif type=='val':
            files=csv.reader(open(args.val_label, 'r'))
            # self.root = args.val_img
            self.root_baseline=args.val_img_baseline
            # self.root_baseline_label=args.val_img_baseline_label
        else:
            print("error+++++++++")


        next(files) 

        self.labels = [label for label in files]
        self.loader = default_loader
        self.trans_img = transforms.Compose([
        transforms.ToTensor()]) 

    def __getitem__(self, index):
        name, hb_vlaue,hb_class = self.labels[index]
        img_name=name+'_0.jpg'
        label_name=name+'_1.png'
        
        hb_class=int(hb_class)
        if hb_class<=9:
            hb_c= 0
        # elif 10<hb_class<=13:
        #     hb_c= 1
        else:
            hb_c= 1
        
        

        # img = Image.open(img).convert('RGB')
        # imgs = [img, img.transpose(Image.FLIP_LEFT_RIGHT)]

   

        # img = self.loader(os.path.join(self.root, img_name))
        img_baseline=self.loader(os.path.join(self.root_baseline, img_name))
        # img_baseline_label=self.loader(os.path.join(self.root_baseline_label, label_name))

        # img_baseline_label=img_baseline_label.convert('L')
        # img_baseline_label=self.trans_img(img_baseline_label).unsqueeze(0)
        

        # re=[7,8,9,10,11,12]
        re=[i for i in range(low,high)]
        label = [normal_sampling(hb_vlaue, i) for i in re]
        label = [i if i > 1e-15 else 1e-15 for i in label]
        label = torch.Tensor(label)

        hb_vlaue = np.float32(hb_vlaue)
        hb_c = np.long(hb_c)
        # hb_c=torch.Tensor(hb_c)
        if self.transform is not None:
            # img = self.transform(img)
            img_baseline = self.transform(img_baseline)
            # img_baseline_label= self.transform(img_baseline_label)


        # print(name)
        # print(img.size())
        return img_baseline,label, hb_vlaue,hb_c

    def __len__(self):
        return len(self.labels)


def normal_sampling(mean, label_k, std=2):
    return math.exp(-(float(label_k)-float(mean))**2/(2*std**2))/(math.sqrt(2*math.pi)*std)


if __name__ == "__main__":
    """Testing
    """
    re=[i for i in range(1,21)]
    r=np.array(re)
    print(re)
    for j in np.arange(6, 18, 0.1):
        plt.figure(1)
        label = [normal_sampling(j, i) for i in re]
        label = [i if i > 1e-15 else 1e-15 for i in label]
        l=np.array(label)
        
        print(label)
        print(np.sum(l*r))
        plt.plot(re,label)
        plt.title(j)
        plt.show()
