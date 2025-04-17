import os

import time
import numpy as np
from skimage import io
import time

import torch, gc
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from data_loader_cache3 import get_im_gt_name_dict, create_dataloaders, GOSRandomHFlip, GOSResize, GOSRandomCrop, GOSNormalize #GOSDatasetCache,
from basics import  f1_mae_torch #normPRED, GOSPRF1ScoresCache,f1score_torch,
from models import *

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
fea_loss = nn.MSELoss(size_average=True)

def structure_loss(pred, mask):

    # BCE loss
    k = nn.Softmax2d()
    weit = torch.abs(pred-mask)
    weit = k(weit)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3))/weit.sum(dim=(2, 3))

    # IOU loss
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1-(inter+1)/(union-inter+1)

    return (wbce + wiou).mean()


def valid(net, valid_dataloaders, valid_datasets, hypar, epoch=0):

    net.to(device)

    net.eval()
    print("Validating...")
    epoch_num = hypar["max_epoch_num"]

    val_loss = 0.0
    tar_loss = 0.0
    val_cnt = 0.0

    tmp_f1 = []
    tmp_mae = []
    tmp_time = []

    start_valid = time.time()

    for k in range(len(valid_dataloaders)):

        valid_dataloader = valid_dataloaders[k]
        valid_dataset = valid_datasets[k]

        val_num = valid_dataset.__len__()
        mybins = np.arange(0,256)
        PRE = np.zeros((val_num,len(mybins)-1))
        REC = np.zeros((val_num,len(mybins)-1))
        F1 = np.zeros((val_num,len(mybins)-1))
        MAE = np.zeros((val_num))

        for i_val, data_val in enumerate(valid_dataloader):
            val_cnt = val_cnt + 1.0
            imidx_val, inputs_val, labels_val, masks_val ,shapes_val = data_val['imidx'], data_val['image'], data_val['label'], data_val['mask'], data_val['shape']

            if(hypar["model_digit"]=="full"):
                inputs_val = inputs_val.type(torch.FloatTensor)
                labels_val = labels_val.type(torch.FloatTensor)
                masks_val = masks_val.type(torch.FloatTensor)
            else:
                inputs_val = inputs_val.type(torch.HalfTensor)
                labels_val = labels_val.type(torch.HalfTensor)
                masks_val = masks_val.type(torch.HalfTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_val_v, labels_val_v, masks_val_v = Variable(inputs_val.cuda(), requires_grad=False), Variable(labels_val.cuda(), requires_grad=False),Variable(masks_val.cuda(), requires_grad=False)
            else:
                inputs_val_v, labels_val_v, masks_val_v = Variable(inputs_val, requires_grad=False), Variable(labels_val,requires_grad=False), Variable(masks_val,requires_grad=False)

            inputs_val_v = inputs_val_v.to(device)
            labels_val_v = labels_val_v.to(device)
            masks_val_v = masks_val_v.to(device)

            t_start = time.time()
            P1,C3, R1, R2, R3= net(inputs_val_v,masks_val_v)
            ds_val=P1
            
            loss1=structure_loss(C3,labels_val_v)
            loss2=structure_loss(R1,labels_val_v)
            loss3=structure_loss(R2,labels_val_v)
            loss4=structure_loss(R3,labels_val_v)
            loss5=structure_loss(P1,labels_val_v)
            loss= loss1 + loss2 + loss3 + loss4 +  loss5

            t_end = time.time()-t_start
            tmp_time.append(t_end)
            ds_val=P1
            loss2_val=loss9
            loss_val=loss

            for t in range(hypar["batch_size_valid"]):
                i_test = imidx_val[t].data.numpy()

                pred_val = ds_val[t,:,:,:] # B x 1 x H x 
                pred_val = torch.squeeze(F.upsample(torch.unsqueeze(pred_val,0),(shapes_val[t][0],shapes_val[t][1]),mode='bilinear'))
                pred_val=torch.sigmoid(pred_val)
                ma = torch.max(pred_val)
                mi = torch.min(pred_val)
                pred_val = (pred_val-mi)/(ma-mi) # max = 1

                if len(valid_dataset.dataset["ori_gt_path"]) != 0:
                    gt = np.squeeze(io.imread(valid_dataset.dataset["ori_gt_path"][i_test])) # max = 255
                    if gt.max()==1:
                        gt=gt*255
                else:
                    gt = np.zeros((shapes_val[t][0],shapes_val[t][1]))
                with torch.no_grad():
                    gt = torch.tensor(gt).to(device)

                pre,rec,f1,mae = f1_mae_torch(pred_val*255, gt, valid_dataset, i_test, mybins, hypar)
                PRE[i_test,:]=pre
                REC[i_test,:] = rec
                F1[i_test,:] = f1
                MAE[i_test] = mae

                del ds_val, gt
                gc.collect()
                torch.cuda.empty_cache()
            val_loss += loss_val.item()#data[0]
            tar_loss += loss2_val.item()#data[0]
            print("[validating: %5d/%5d] val_ls:%f, tar_ls: %f, f1: %f, mae: %f, time: %f"% (i_val, val_num, val_loss / (i_val + 1), tar_loss / (i_val + 1), np.amax(F1[i_test,:]), MAE[i_test],t_end))
            del loss2_val, loss_val

        print('============================')
        PRE_m = np.mean(PRE,0)
        REC_m = np.mean(REC,0)
        f1_m = (1+0.3)*PRE_m*REC_m/(0.3*PRE_m+REC_m+1e-8)

        tmp_f1.append(np.amax(f1_m))
        tmp_mae.append(np.mean(MAE))

    return tmp_f1, tmp_mae, val_loss, tar_loss, i_val, tmp_time

def main(train_datasets,
         valid_datasets,
         hypar):
    dataloaders_train = []
    dataloaders_valid = []

    if(hypar["mode"]=="train"):
        print("--- create training dataloader ---")
        train_nm_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
        train_dataloaders, train_datasets = create_dataloaders(train_nm_im_gt_list,
                                                             cache_size = hypar["cache_size"],
                                                             cache_boost = hypar["cache_boost_train"],
                                                             my_transforms = [
                                                                             GOSRandomHFlip(), ## this line can be uncommented for horizontal flip augmetation
                                                                             # GOSResize(hypar["input_size"]),
                                                                             # GOSRandomCrop(hypar["crop_size"]), ## this line can be uncommented for randomcrop augmentation
                                                                              GOSNormalize([0.5,0.5,0.5],[1.0,1.0,1.0]),
                                                                              ],
                                                             batch_size = hypar["batch_size_train"],
                                                             shuffle = True)
        print("sucessfully load")
        train_dataloaders_val, train_datasets_val = create_dataloaders(train_nm_im_gt_list,
                                                             cache_size = hypar["cache_size"],
                                                             cache_boost = hypar["cache_boost_train"],
                                                             my_transforms = [
                                                                              GOSNormalize([0.5,0.5,0.5],[1.0,1.0,1.0]),
                                                                              ],
                                                             batch_size = hypar["batch_size_valid"],
                                                             shuffle = False)
        print(len(train_dataloaders), " train dataloaders created")

    print("--- create valid dataloader ---")
    valid_nm_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    valid_dataloaders, valid_datasets = create_dataloaders(valid_nm_im_gt_list,
                                                          cache_size = hypar["cache_size"],
                                                          cache_boost = hypar["cache_boost_valid"],
                                                          my_transforms = [
                                                                           GOSNormalize([0.5,0.5,0.5],[1.0,1.0,1.0]),
                                                                           # GOSResize(hypar["input_size"])
                                                                           ],
                                                          batch_size=hypar["batch_size_valid"],
                                                          shuffle=False)
    print(len(valid_dataloaders), " valid dataloaders created")
    ### --- Step 2: Build Model and Optimizer ---
    print("--- build model ---")
    net = hypar["model"]#GOSNETINC(3,1)

    # convert to half precision
    if(hypar["model_digit"]=="half"):
        net.half()
        for layer in net.modules():
          if isinstance(layer, nn.BatchNorm2d):
            layer.float()

    if torch.cuda.is_available():
        net.cuda()

    if(hypar["restore_model"]!=""):
        print("restore model from:")
        print(hypar["model_path"]+"/"+hypar["restore_model"])
        if torch.cuda.is_available():
            checkpoint=torch.load(hypar["model_path"]+"/"+hypar["restore_model"])
            model_dict = net.state_dict()
            pretrained_dict = {k.split('module.')[-1]: v for k, v in checkpoint.items() if (k.split('module.')[-1] in model_dict)}
            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict)

            net.cuda()
        else:
            net.load_state_dict(torch.load(hypar["model_path"]+"/"+hypar["restore_model"],map_location="cpu"))

    print("--- define optimizer ---")
    optimizer = torch.optim.AdamW(net.parameters(), 1e-5, weight_decay=1e-4)
    ### --- Step 3: Train or Valid Model ---
    if(hypar["mode"]=="train"):
        train(net,
              optimizer,
              train_dataloaders,
              train_datasets,
              valid_dataloaders,
              valid_datasets,
              hypar,
              train_dataloaders_val, train_datasets_val)
    else:
        valid(net,
              valid_dataloaders,
              valid_datasets,
              hypar)


if __name__ == "__main__":

    train_datasets, valid_datasets = [], []
    dataset_1, dataset_1 = {}, {}

    dataset_tr = {"name": "DIS5K-TR",
                 "im_dir": "",
                 "gt_dir": "",
                 "mask_dir": "",
                 "im_ext": ".jpg",
                 "gt_ext": ".png",
                 "mask_ext": ".png",
                 "cache_dir":""}

    dataset_vd = {"name": "DIS5K-VD",
                 "im_dir": "",
                 "gt_dir": "",
                 "mask_dir": "",
                 "im_ext": ".jpg",
                 "gt_ext": ".png",
                 "mask_ext": ".png",
                 "cache_dir":""}

    dataset_te1 = {"name": "DIS5K-TE1",
                 "im_dir": "",
                 "gt_dir": "",
                 "mask_dir": "",
                 "im_ext": ".jpg",
                 "gt_ext": ".png",
                 "mask_ext": ".png",
                 "cache_dir":""}

    dataset_te2 = {"name": "DIS5K-TE2",
                 "im_dir": "",
                 "gt_dir": "",
                 "mask_dir": "",
                 "im_ext": ".jpg",
                 "gt_ext": ".png",
                 "mask_ext": ".png",
                 "cache_dir":""}

    dataset_te3 = {"name": "DIS5K-TE3",
                 "im_dir": "",
                 "gt_dir": "",
                 "mask_dir": "",
                 "im_ext": ".jpg",
                 "gt_ext": ".png",
                 "mask_ext": ".png",
                 "cache_dir":""}

    dataset_te4 = {"name": "DIS5K-TE4",
                 "im_dir": "",
                 "gt_dir": "",
                 "mask_dir": "",
                 "im_ext": ".jpg",
                 "gt_ext": ".png",
                 "mask_ext": ".png",
                 "cache_dir":""}
    dataset_demo = {"name": "DIS-testvd",
                 "im_dir": "",
                 "gt_dir": "",
                 "mask_dir": "",
                 "im_ext": ".jpg",
                 "gt_ext": ".png",
                 "mask_ext": ".png",
                 "cache_dir":""}

    train_datasets = [dataset_tr] 
    valid_datasets = [dataset_vd, dataset_te1, dataset_te2, dataset_te3, dataset_te4] 

    hypar = {}
    hypar["mode"] = "valid"
    hypar["interm_sup"] = False 

    if hypar["mode"] == "valid":
        hypar["valid_out_dir"] = "/home/fabian/BRL/zhoushanfeng/2024GoodNet/Results/goodnet-RFE+CFR+CSAM_predsr+blinear_outSAM_NCD4000"
        hypar["model_path"] = "/home/fabian/BRL/zhoushanfeng/2024GoodNet/saved_models/piror_mask/good_result_net-RFE+CFR+CSAM_predsr+blinear_outSAM_NCD/" 
        hypar["restore_model"] = "gpu_itr_4000_traLoss_0.4197_traTarLoss_0.0658_valLoss_7.6549_valTarLoss_1.4329_maxF1_0.8983_0.8975_0.8925_mae_0.0391_0.0356_0.0484_time_0.145939.pth"

    hypar["model_digit"] = "full" 
    hypar["seed"] = 0

    hypar["cache_size"] = [1024, 1024] 
    hypar["cache_boost_train"] = False 


    hypar["input_size"] = [1024, 1024] 
    hypar["crop_size"] = [1024, 1024] 
    hypar["random_flip_h"] = 1 
    hypar["random_flip_v"] = 0 


    print("building model...")
    hypar["model"] = goodnet() 
    hypar["early_stop"] = 40 
    hypar["model_save_fre"] = 3000 

    hypar["batch_size_train"] = 4 
    hypar["batch_size_valid"] = 1 
    print("batch size: ", hypar["batch_size_train"])

    hypar["max_ite"] = 400000 
    hypar["max_epoch_num"] = 1000000 

    main(train_datasets,
         valid_datasets,
         hypar=hypar)