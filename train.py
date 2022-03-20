'''
Author: Linmin Pei
Date: Nov. 10th, 2020
'''
from configparser import ConfigParser
import os
import sys
import time
import numpy as np
import configparser
import torch
from torchvision import transforms, utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.unet_ensemble import UNet
import torch.optim as optim
from utils.datasets import *
from utils import criterion
from utils.calDice import cal_dice
import random

print('....torch version: ', torch.__version__)
sPhase = 'valid'  # 'train', 'valid', or 'test'
cFile = 'config.ini'  # configuration file
config = ConfigParser()
config.read(cFile)

root_dir = config['DEFAULT']['root_dir']  # root dir
criterionMethod = config['DEFAULT']['criterion']  # loss criterion
norm = config['DEFAULT']['norm']  # normalization type
bResume = int(config['DEFAULT']['bResume'])  # Resume model
num_ndf = int(config['DEFAULT']['num_ndf'])  # number of ndf
s_modality = config['DEFAULT']['s_modality']  # number of channel
num_class = int(config['DEFAULT']['num_class'])  # number of class
num_epoch = int(config['DEFAULT']['num_epoch'])  # number of epoch
num_batch = int(config['DEFAULT']['batch_size'])  # number of batch size
num_step = int(config['DEFAULT']['nStep'])  # each nStep epoch to save a model
bTradition = eval(config['DEFAULT']['bTradition'])
num_frequence = int(config['DEFAULT']['nFrequence']
                    )  # validate in every n epoch
eps = float(config['DEFAULT']['eps'])
seed = int(config['DEFAULT']['seed'])  # seed
LR = float(config['DEFAULT']['lr'])  # learning rate (default: 0.001)
sTrans_train = config['DEFAULT']['transforms']
model_path = config['DEFAULT']['model_path']  # seed
train_list = config[sPhase]['train_list']  # train list
valid_list = config[sPhase]['valid_list']  # valid list
img_dir = config[sPhase]['img_dir']  # image dir
num_channel = len(s_modality)
print('..modality: {} in {} phase '.format(s_modality, sPhase))

cwd = os.getcwd()
saveDiceResult = os.path.join(cwd, 'log_train_'+s_modality+'.txt')
if os.path.exists(saveDiceResult) == True:
    os.remove(saveDiceResult)

bGPU = torch.cuda.is_available()  # check gpu
modelDir = os.path.join(root_dir, model_path)  # path to save models
if os.path.exists(modelDir) == False:
    # create a directory to save trained model
    os.makedirs(modelDir)

train_dataset = DriveData(
    root_dir, img_dir, train_list, sTrans_train, s_modality)
train_loader = DataLoader(train_dataset, num_batch, collate_fn=train_dataset.collate, num_workers=8,
                          pin_memory=True, shuffle=True)
if sPhase == 'valid':
    # transformation for validation
    sTrans_valid = config[sPhase]['transforms']
    valid_dataset = DriveData(
        root_dir, img_dir, valid_list, sTrans_valid, s_modality)
    valid_loader = DataLoader(valid_dataset, num_batch, collate_fn=train_dataset.collate, num_workers=8,
                              pin_memory=True, shuffle=False)


model = UNet(num_channel, num_class, num_ndf)
if bGPU == True:  # gpu use
    model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=LR)
if bResume == 1:
    CKPT_file = os.path.join(modelDir, 'ckpt_300')
    checkpoint = torch.load(CKPT_file)
    nStart_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("===> loaded ckpt at (epoch {}) from {}".format(
        checkpoint['epoch'], CKPT_file))
else:
    nStart_epoch = 0

# criterionLoss = getattr(criterion, criterionMethod)
criterionLoss = torch.nn.BCELoss()
# convert subregion label to regular mask: 3x155x240x240-->155x240x240


def initial(seed):  # initialization
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))


def adjust_learning_rate(optimizer, LR, epoch):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = LR * ((1-epoch/num_epoch).__pow__(0.9))
        param_group['lr'] = lr
        print('...working on epoch: {}/{} with learning rate: {:.7f}'.format(epoch,
              num_epoch, param_group['lr']))


def train():
    model.train()  # change to training mode
    factor = 0.00001
    total_loss = 0
    total_dice = 0
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.type(
            torch.FloatTensor), target.type(torch.FloatTensor)
        data, target = Variable(data), Variable(target)  # gpu version
        if bGPU == True:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterionLoss(output, target)

        reg_loss = None
        for param in model.parameters():
            if reg_loss is None:
                reg_loss = param.norm(2)
            else:
                reg_loss = reg_loss + param.norm(2)
        loss = loss + factor*reg_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.data.cpu().numpy()
        dice = cal_dice(output, target, bGPU, num_batch, bTradition)
        total_dice += dice

    ave_loss = total_loss/len(train_loader.dataset)
    ave_dice = total_dice/len(train_loader.dataset)
    return ave_loss, ave_dice


def validation():
   # run validation in every num_frequence epoch
    model.eval()  # change to testing mode
    total_loss = 0
    total_dice = 0
    for idx, (data, target) in enumerate(valid_loader):
        data, target = data.type(
            torch.FloatTensor), target.type(torch.FloatTensor)
        data, target = Variable(data), Variable(target)  # gpu version
        if bGPU == True:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = model(data)
        loss = criterionLoss(output, target)

        total_loss += loss.data.cpu().numpy()
        dice = cal_dice(output, target, bGPU, num_batch)
        total_dice += dice
    ave_loss = total_loss/len(valid_loader.dataset)
    ave_dice = total_dice/len(valid_loader.dataset)

    return ave_loss, ave_dice


def main():
    initial(seed)
    nLoss, nDice = 10000, 0  # initial loss
    tempLoss, tempDice = 0, 0

    fd = open(saveDiceResult, "a")
    fd.write(str('Epoch')+"\t\t"+str('Loss')+"\t"+str('Dice')+"\n")
    for epoch in range(nStart_epoch, num_epoch):
        adjust_learning_rate(optimizer, LR, epoch)  # adjust lr
        #print('...working on epoch: {}/{} with learing rate: {:.6f}'.format(epoch, num_epoch, temp_lr))

        loss_train, dice_train = train()
        print('...training: {:.4f}, {:.4f}.'.format(loss_train, dice_train))
        if np.logical_and(sPhase == 'valid', epoch % num_frequence == 0):
            loss_valid, dice_valid = validation()
            print('+++valid: {:.4f}, {:.4f}.'.format(loss_valid, dice_valid))
        fd.write(str(epoch+1)+"\t\t"+str(np.round(nLoss, 6)) +
                 "\t"+str(np.round(nDice, 4))+"\n")

        if sPhase == 'train':
            tempLoss, tempDice = loss_train, dice_train
        elif sPhase == 'valid':
            tempLoss, tempDice = loss_valid, dice_valid
        else:
            raise ValueError('Unknown phase value!')

        if tempDice > nDice:
            print('*****phase: %s: updating dice from: %.4f, to: %.4f' %
                  (sPhase, nDice, tempDice))
            nLoss, nDice = tempLoss, tempDice   # update the loss and dice
            torch.save(model, os.path.join(
                modelDir, 'bestModel_' + s_modality))
        # elif np.mod(epoch, nStep) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(modelDir, 'ckpt_' + s_modality))
        print('------------------------------------------------------------\n')
    fd.close()


if __name__ == '__main__':
    startTime = time.time()
    main()
    endTime = time.time()
    print('It takes %d seconds to complete!' % (endTime-startTime))
