#!/user/bin/python
# coding=utf-8
import numpy as np
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler, optimizer
from models.CAPS_models.unet_lps.unet_model import *
#from models.aps_models.unet_aps.unet_model import *
import cv2 as cv
from torch.utils.data import DataLoader, sampler,Dataset
import random
from utils.dice_score import dice_loss
from evaluate import evaluate
from torch.utils.tensorboard import SummaryWriter
load_model = False
n_classes = 2
writer = SummaryWriter('runs')

print("the device is:",device)
file1 = open('./result_CAPS_youwu1.txt','a+')
train_mode = 1
def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

class SegmentationDataset(Dataset):
    def __init__(self, img_dir,mask_dir,mode):
        self.images = img_dir
        self.masks = mask_dir
    def __len__(self):
        return len(self.images)

    def num_of_samples(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            image_path = self.images[idx]
            mask_path = self.masks[idx]
        else:
            image_path = self.images[idx]
            mask_path = self.masks[idx]
        img = cv.imread(image_path, cv.IMREAD_COLOR)  # BGR order
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        # input image
        img = np.float32(img) / 255.0
        img = img.transpose((2,0,1))
        mask[mask <= 0] = 0  
        mask[mask > 0] = 1    
        sample = {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask),}
        return sample

if __name__ == '__main__':
    index = 0
    num_epochs = 200
    train_on_gpu = True
    unet = UNet_4down_CAPS(in_channels=3, out_channels=2,padding_mode = 'zeros',comp_train_convex=True,comp_convex=False).to(device)
    if load_model:
        if device.type=="cpu":
            model_dict=unet.load_state_dict(torch.load('unet_epoch.pt',map_location=torch.device('cpu')))
        else:
            model_dict=unet.load_state_dict(torch.load('unet_epoch.pt'))
    
    image_train_dir = './image_train/'
    mask_train_dir = './mask_train/'
    image_test_dir = './image_test/'
    mask_test_dir = './mask_test/'
    
    images_path_list = []
    masks_path_list = []
    test_img_list = []
    test_mask_list = []
    files = os.listdir(image_train_dir)
    sfiles = os.listdir(image_test_dir)
    for i in range(len(files)):
        img_file = os.path.join(image_train_dir, files[i])
        number = os.path.basename(img_file).split('.')[0]
        mask_file = os.path.join(mask_train_dir, number+".png")
        # print(img_file, mask_file)
        images_path_list.append(img_file)
        masks_path_list.append(mask_file)

    for i in range(len(sfiles)):
        img_file = os.path.join(image_test_dir, sfiles[i])
        number = os.path.basename(img_file).split('.')[0]
        mask_file = os.path.join(mask_test_dir, number+".png")
        # print(img_file, mask_file)
        test_img_list.append(img_file)
        test_mask_list.append(mask_file)
    

    zhen = list(zip(images_path_list,masks_path_list))
    random.seed(228)
    random.shuffle(zhen)
    images_path_list[:], masks_path_list[:] = zip(*zhen)
    print(images_path_list[0],masks_path_list[0])
    print(images_path_list[1],masks_path_list[1])
    num_of_samples = len(images_path_list)

    if train_mode == 1:
    
        dataloader_train = SegmentationDataset(images_path_list[:int(0.8*num_of_samples)], masks_path_list[:int(0.8*num_of_samples)],mode = "train")
        train_loader = DataLoader(
            dataloader_train, batch_size=64, shuffle=False)
        dataloader_test = SegmentationDataset(images_path_list[int(0.8*num_of_samples):], masks_path_list[int(0.8*num_of_samples):],mode = "val")
        test_loader = DataLoader(dataloader_test,batch_size=16,shuffle=False,drop_last=True)
    else:
        dataloader_train = SegmentationDataset(images_path_list[:], masks_path_list[:],mode = "train")
        train_loader = DataLoader(
            dataloader_train, batch_size=64, shuffle=False)
        dataloader_test = SegmentationDataset(test_img_list[:], test_mask_list[:],mode = "val")
        test_loader = DataLoader(dataloader_test,batch_size=16,shuffle=False,drop_last=True)
    weight_decay = True
    filter_bias_and_bn = True
    weight_decay = 1e-4
    if weight_decay and filter_bias_and_bn:
        parameters = add_weight_decay(unet, weight_decay)
        weight_decay = 0.
    else:
        parameters = unet.parameters()
    optimizer = torch.optim.SGD(parameters, lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',factor = 0.8,patience=3,min_lr = 0.00005)
    grad_scaler = torch.cuda.amp.GradScaler(enabled= False)
    criterion = nn.CrossEntropyLoss()
    iou_temp = 0
    file1.write('===============================================================================\n')
    for epoch in range(num_epochs):
        unet.train()
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        lr_now = optimizer.state_dict()['param_groups'][0]['lr']
        train_loss = 0.0
        for i_batch, sample_batched in enumerate(train_loader):
            images_batch, target_labels = \
                sample_batched['image'], sample_batched['mask']
            if train_on_gpu:
                images_batch, target_labels = images_batch.to(device, dtype=torch.float32), target_labels.to(device,dtype=torch.long)
            
            with torch.cuda.amp.autocast(enabled=False):
                masks_pred = unet(images_batch)
                loss = criterion(masks_pred, target_labels) \
                        + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                    F.one_hot(target_labels, n_classes).permute(0, 3, 1, 2).float(),
                                    multiclass=True)
                
            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            train_loss += loss.item()
            print('step: {}  current Loss: {:.6f} '.format(index, loss.item()))
            index += 1
            writer.add_scalar('train_loss',loss.item(),global_step=index)
        train_loss = train_loss / dataloader_train.num_of_samples()
        val_score,val_loss = evaluate(unet, test_loader,dataloader_test, device)
        scheduler.step(val_score)
        writer.add_scalar('val_loss', val_loss.item(), global_step=epoch)
        writer.add_scalar('val_dice', val_score.item(), global_step=epoch)
        print('Epoch: {}   Training Loss: {:.6f} '.format(epoch, train_loss))
        print('Epoch: {}   Testing Loss: {:.6f}    dice: {:.6f}  '.format(epoch, val_loss.item(),val_score.item()))
        file1.write(f'Epoch: {epoch}   Testing Loss: {val_loss.item()}   dice:{val_score.item()}   lr:{lr_now}'+'\n')
        file1.flush()  
        if val_score > iou_temp:
            torch.save(unet.state_dict(), 'unet_epoch_CAPS_youwu1.pt')
            iou_temp = val_score
            file1.write(f'save best model :Epoch: {epoch} dice:{val_score.item()} '+'\n')
        if epoch % 50 == 0 and epoch>50:
            torch.save(unet.state_dict(), 'unet_youwu1_CAPS_%d.pt'%epoch)
    unet.eval()
    file1.close()
    print(iou_temp)
    print("begin test")
    dataloader_test = SegmentationDataset(test_img_list[:], test_mask_list[:],mode = "val")
    test_loader = DataLoader(dataloader_test,batch_size=16,shuffle=False,drop_last=True)
    val_score,val_loss = evaluate(unet, test_loader,dataloader_test, device)
    print(val_score.item())
    print("end test")


    

