'''
Author: Linmin Pei
Date: Nov. 24th, 2020
Difference between test_raw.py and test_online.py is the data saved form.
In test_online.py, all image modalities are fused into one nifti file.
However, in test_raw.py, all image modalities are saved as a seperated files (even in raw image status)
'''
from configparser import ConfigParser
import os
import time
import numpy as np
import configparser
import torch
from torchvision import transforms, utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.unet_ensemble import UNet
import torch.optim as optim
from utils import mask_conversion
from utils.datasets import *
from utils import criterion
from utils.calDice import cal_dice
from utils.post_processing import post_processing
from utils.loadFusionImage import loadFusionImage
from utils.imagePadding import *
import SimpleITK as sitk
import random
# import ipdb
# ipdb.set_trace(context=20)

sPhase = 'test'  # 'train', 'valid', or 'test'
cFile = 'config.ini'  # configuration file
all_dic = {'f': '_flair.nii.gz', '1': '_t1.nii.gz',
           'c': '_t1ce.nii.gz', '2': '_t2.nii.gz'}
modality = ['_flair.nii.gz', '_t1.nii.gz', '_t1ce.nii.gz', '_t2.nii.gz']
bNorm = True  # True: need to be normalized, otherwise False
bPostProcess = True
bTTA = True
# raw, or fusion. raw: separate raw image in a folder. fusion: a fused file
sDataType = 'raw'

config = ConfigParser()
config.read(cFile)

root_dir = config['DEFAULT']['root_dir']  # root dir
num_ndf = int(config['DEFAULT']['num_ndf'])  # number of ndf
num_batch = int(config['DEFAULT']['batch_size'])  # number of batch size
s_modality = config['DEFAULT']['s_modality']  # number of channel
num_class = int(config['DEFAULT']['num_class'])  # number of class
seed = int(config['DEFAULT']['seed'])  # seed
bTradition = eval(config['DEFAULT']['bTradition'])
model_path = config['DEFAULT']['model_path']  # seed
img_dir = config[sPhase]['img_dir']  # image dir
seg_dir = config[sPhase]['seg_dir']  # root dir
num_channel = len(s_modality)
modality = [all_dic[x] for x in s_modality]

sModelName = 'ckpt_'+s_modality

# bGPU = False
bGPU = torch.cuda.is_available()  # check gpu
modelDir = os.path.join(root_dir, model_path)  # path to save models
maskConversionObj = getattr(mask_conversion, 'de_conversion')


def loadModel(sModelName):
    model = UNet(num_channel, num_class, num_ndf)
    # model = torch.nn.DataParallel(model)
    if bGPU == True:
        ckpt = torch.load(os.path.join(modelDir, sModelName))
    else:
        ckpt = torch.load(os.path.join(
            modelDir, sModelName), map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt['model_state_dict'])

    if bGPU == True:  # gpu use
        model = model.cuda()
    return model


def initial(seed):  # initialization
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))


imageDir = os.path.join(root_dir, img_dir)
# segDir = os.path.join(root_dir, 'segResults')
model = loadModel(sModelName)


def execution(patList):
    model.eval()  # change to testing mode
    for idx, pid in enumerate(patList):
        if sDataType == 'raw':
            print('...working on {}: {}/{}'.format(pid, idx+1, len(patList)))
            pid_path = os.path.join(imageDir, pid)  # get image full path
            dataInfo, data = loadFusionImage(pid_path, pid, modality, bNorm)
            origin = dataInfo['origin']
            spacing = dataInfo['spacing']
            direction = dataInfo['direction']
        elif sDataType == 'fusion':
            pid_path = os.path.join(imageDir, pid)  # get image full path
            pid = pid[:-7]  # only excluse .nii.gz
            print('...working on {}: {}/{}'.format(pid, idx+1, len(patList)))
            pid_obj = sitk.ReadImage(pid_path)
            origin = pid_obj.GetOrigin()
            spacing = pid_obj.GetSpacing()
            direction = pid_obj.GetDirection()
            data = sitk.GetArrayFromImage(pid_obj)  # get data: 155x240x240x4
            data = np.transpose(data, (3, 0, 1, 2))  # change to: 4x155x240x240
        else:
            raise ValueError('Error, undefined data type')

        if num_channel == 2:
            data = np.delete(data, 1, 0)  # delete t1 data
            data = np.delete(data, 2, 0)  # delete t2 data
        elif num_channel == 3:
            data = np.delete(data, 3, 0)  # delete t2 data only
        elif num_channel == 4:
            pass
        else:
            raise ValueError('xxx: channel number is wrong!')

        # data shape: 4x155x240x240data = imgPadding(data)
        [nSlice, nRow, nCol] = data.shape[1:]
        data = imgPadding(data, 16)  # image padding

        # after normalization, pix value in background is around -3
        data = data[None, ...]  # 1x4x160x240x240
        data = torch.from_numpy(data)

        data = data.type(torch.FloatTensor)
        data = Variable(data)  # gpu version
        if bGPU == True:
            data = data.cuda()
        with torch.no_grad():
            output = model(data)

            if bTTA == True:
                output += model(data.flip(dims=(2, ))).flip(dims=(2,))
                output += model(data.flip(dims=(3, ))).flip(dims=(3,))
                output += model(data.flip(dims=(4, ))).flip(dims=(4,))
                output += model(data.flip(dims=(2, 3))).flip(dims=(2, 3))
                output += model(data.flip(dims=(2, 4))).flip(dims=(2, 4))
                output += model(data.flip(dims=(3, 4))).flip(dims=(3, 4))
                output += model(data.flip(dims=(2, 3, 4))).flip(dims=(2, 3, 4))
                output = output / 8.0
            prediction = output[0]  # batchsize = 1
            if bGPU == True:
                prediction_arr = prediction.data.cpu().numpy()
            else:
                prediction_arr = prediction.data.numpy()
            seg = (prediction_arr > 0.5).astype(np.int)[0]
            # seg = maskConversionObj(prediction_arr, bTradition)
            # seg = seg[:155]  # crop to 155x240x240 because of padding

        if bPostProcess == True:
            seg = post_processing(seg)

        # de-padding if there is padding before
        seg = deImgPadding(seg, nRow, nCol, nSlice).astype(np.uint8)

        # if os.path.exists(segDir) == False:
        #     os.mkdir(segDir)
        savedName = pid+'_mask.nii.gz'

        seg_obj = sitk.GetImageFromArray(seg)
        seg_obj.SetOrigin(origin)
        seg_obj.SetSpacing(spacing)
        seg_obj.SetDirection(direction)
        segDir = os.path.join(pid_path, seg_dir)

        sitk.WriteImage(seg_obj, os.path.join(pid_path, savedName))


def main(model):
    patList = os.listdir(imageDir)
    patList.sort()
    initial(seed)
    execution(patList)


if __name__ == '__main__':
    startTime = time.time()
    main(model)
    endTime = time.time()
    print('It takes %d seconds to complete!' % (endTime-startTime))
