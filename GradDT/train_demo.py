# -*- coding: utf-8 -*-
"""
Created on Sat May 16 11:01:08 2020

@author: Administrator
"""

import os, time, scipy.io, shutil
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import cv2
import scipy.misc
from model import *
from makedataset import Dataset
import utils_train
import SSIM
from utils.noise import *
from utils.common import *
from utils.loss import *
import pandas as pd


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_checkpoint(checkpoint_dir, num_input_channels):
    if num_input_channels ==3:
        
        if os.path.exists(checkpoint_dir + 'checkpoint.pth.tar'):
            # load existing model
            model_info = torch.load(checkpoint_dir + 'checkpoint.pth.tar')
            print('==> loading existing model:', checkpoint_dir + 'checkpoint.pth.tar')
            net = DSPNet()
            device_ids = [0]
            model = nn.DataParallel(net, device_ids=device_ids).cuda()
            model.load_state_dict(model_info['state_dict'])
            optimizer = torch.optim.Adam(model.parameters())
            optimizer.load_state_dict(model_info['optimizer'])
            cur_epoch = model_info['epoch']
            
        else:
            # create model
            net = DSPNet()
            device_ids = [0]
            model = nn.DataParallel(net, device_ids=device_ids).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            cur_epoch = 0

    else:
        if os.path.exists(checkpoint_dir + 'checkpoint.pth.tar'):
            # load existing model
            model_info = torch.load(checkpoint_dir + 'checkpoint.pth.tar')
            print('==> loading existing model:', checkpoint_dir + 'checkpoint.pth.tar')
            net = DSPNet()
            device_ids = [0]
            model = nn.DataParallel(net, device_ids=device_ids).cuda()
            model.load_state_dict(model_info['state_dict'])
            optimizer = torch.optim.Adam(model.parameters())
            optimizer.load_state_dict(model_info['optimizer'])
            cur_epoch = model_info['epoch']
            
        else:
            # create model
            net = DSPNet()
            device_ids = [0]
            model = nn.DataParallel(net, device_ids=device_ids).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            cur_epoch = 0


    return model, optimizer,cur_epoch

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	torch.save(state, checkpoint_dir + 'checkpoint.pth.tar')
	if is_best:
		shutil.copyfile(checkpoint_dir + 'checkpoint.pth.tar',checkpoint_dir + 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch, lr_update_freq):
	if not epoch % lr_update_freq and epoch:
		for param_group in optimizer.param_groups:
			param_group['lr'] = param_group['lr'] * 0.1
			print( param_group['lr'])
	return optimizer

def train_psnr(log_train_in,log_train_out):
    train_in = torch.exp(-log_train_in)
    train_out = torch.exp(-log_train_out)
    psnr = utils_train.batch_psnr(train_in,train_out,1.)
    return psnr

def FSobel_X(data):
    x = cv2.Sobel(np.clip(data*255,0,255).astype('uint8'),cv2.CV_8U,0,1,ksize =3)
    absX = cv2.convertScaleAbs(x)
    return absX/255.0
 
def FSobel_Y(data):
    x = cv2.Sobel(np.clip(data*255,0,255).astype('uint8'),cv2.CV_8U,1,0,ksize =3)
    absY = cv2.convertScaleAbs(x)
    return absY/255.0

def FLap(data):
    x = cv2.Laplacian(np.clip(data*255,0,255).astype('uint8'),cv2.CV_8U,ksize =3)
    Lap = cv2.convertScaleAbs(x)
    return Lap/255.0

def train_synthetic(data):
 
    maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
    imgo_train =data
    img_train = -torch.log(data+1e-3)
    noiseimg = torch.zeros(img_train.size())

    clearimg_sobelx = torch.zeros(img_train.size())
    noiseimg_sobelx = torch.zeros(img_train.size())
    clearimg_sobely = torch.zeros(img_train.size())
    noiseimg_sobely= torch.zeros(img_train.size())
    clearimg_lap = torch.zeros(img_train.size())
    noiseimg_lap = torch.zeros(img_train.size())
    
    for nx in range(imgo_train.shape[0]):
        noiseimg[nx,:,:,:] = torch.from_numpy(AddRealNoise(img_train[nx, :, :, :].numpy()))

        clearimg_sobelx[nx,0,:,:] = torch.from_numpy(FSobel_X(img_train[nx, 0, :, :].numpy()))
        noiseimg_sobelx[nx,0,:,:] = torch.from_numpy(FSobel_X(noiseimg[nx, 0, :, :].numpy()))

        clearimg_sobely[nx,0,:,:] = torch.from_numpy(FSobel_Y(img_train[nx, 0, :, :].numpy()))
        noiseimg_sobely[nx,0,:,:] = torch.from_numpy(FSobel_Y(noiseimg[nx, 0, :, :].numpy()))

        clearimg_lap[nx,0,:,:] = torch.from_numpy(FLap(img_train[nx, 0, :, :].numpy()))
        noiseimg_lap[nx,0,:,:] = torch.from_numpy(FLap(noiseimg[nx, 0, :, :].numpy()))

        
    input_var = Variable(noiseimg.cuda(), volatile=True)
    target_var = Variable(img_train.cuda(), volatile=True)

    inputsx_var = Variable(noiseimg_sobelx.cuda(), volatile=True)
    targetsx_var = maxpool(maxpool(Variable(clearimg_sobelx.cuda(), volatile=True)))
    
    inputsy_var = Variable(noiseimg_sobely.cuda(), volatile=True)
    targetsy_var = maxpool(maxpool(Variable(clearimg_sobely.cuda(), volatile=True)))
    
    inputlap_var = Variable(noiseimg_lap.cuda(), volatile=True)
    targetlap_var = maxpool(maxpool(Variable(clearimg_lap.cuda(), volatile=True)))
    
    
    return input_var , target_var , inputsx_var , targetsx_var , inputsy_var , targetsy_var , inputlap_var , targetlap_var

def train_real(data,num_input_channels):
    c = num_input_channels
    imgc_train = -torch.log(data[:,0:c,:,:]+1e-3)
    imgn_train = -torch.log(data[:,c:2*c,:,:]+1e-3)
    noiselevel = torch.zeros(imgc_train.size())
    for nx in range(imgc_train.shape[0]):
        noiselevel[nx,:,:,:] = imgn_train[nx,:,:,:]-imgc_train[nx,:,:,:]
   
    input_var = Variable(imgn_train.cuda(), volatile=True)
    target_var = Variable(imgc_train.cuda(), volatile=True)
    noise_level_var = Variable(noiselevel.cuda(), volatile=True)  
      
    return input_var, target_var, noise_level_var
def load_excel(x,y):
    data1 = pd.DataFrame(x)
    data2 = pd.DataFrame(y)

    writer = pd.ExcelWriter('./A.xlsx')		# 写入Excel文件
    data1.to_excel(writer, '-PSNR', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
    data2.to_excel(writer, '-SSIM', float_format='%.5f')

    writer.save()
    writer.close()
    

def test_synthetic(test_syn,result_syn, model,epoch,num_input_channels,m,n):
    
    noiselevel = [5,10,15,20]
    files = os.listdir(test_syn)
    p=0; s=0
    
    for i in range(len(noiselevel)):
        ssim=0.0; psnr=0.0
        for j in range(len(files)):   
            model.eval()
            with torch.no_grad():
                img_c =  cv2.imread(test_syn + '/' + files[j])
                if num_input_channels == 3:
                    img_cc = img_c[:,:,::-1] / 255.0
                    clear_img = -np.log(img_cc+1e-3)
                    clear_img = np.array(clear_img).astype('float32')
                    noise = np.zeros(clear_img.shape)
                    w,h,c = noise.shape
                    noise_img = clear_img + (-np.log(np.random.gamma(shape=noiselevel[i],scale = 1/noiselevel[i],size =(w,h,c))+1e-3))
                    noise_img_chw = hwc_to_chw(noise_img)
                    input_var = torch.from_numpy(noise_img_chw.copy()).type(torch.FloatTensor).unsqueeze(0)
                    
                    input_var = input_var.cuda()
                    output_16,output_4,output = model(input_var)                         
                    #output_np_4 = chw_to_hwc(output_np_4)                    
                    #output_np_16 = chw_to_hwc(output_np_16)                    

                else:
                    img_cc =  cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY) / 255.0
                    clear_img = -np.log(img_cc+1e-3)
                    clear_img = np.array(clear_img).astype('float32')
                    noise = np.zeros(clear_img.shape)
                    w,h = noise.shape
                    noise_img = clear_img + (-np.log(np.random.gamma(shape=noiselevel[i],scale = 1/noiselevel[i],size =(w,h))+1e-3))

                    true1_var = torch.from_numpy(FSobel_X(clear_img).copy()).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).cuda()
                    true2_var = torch.from_numpy(FSobel_Y(clear_img).copy()).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).cuda()
                    true3_var = torch.from_numpy(FLap(clear_img).copy()).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).cuda()
                    
                    input_var = torch.from_numpy(noise_img.copy()).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).cuda()
                    input_sxvar = torch.from_numpy(FSobel_X(noise_img.copy())).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).cuda()
                    input_syvar = torch.from_numpy(FSobel_Y(noise_img.copy())).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).cuda()
                    input_lapvar = torch.from_numpy(FLap(noise_img.copy())).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).cuda()

                    sxout,syout,lapout,output = model(input_var, input_sxvar, input_syvar, input_lapvar)            

                                 
                output_np = torch.exp(-output)
                output_np= output_np[0,0,:,:].cpu().detach().numpy()
                
                a,b = output_np.shape

                outputsx_np = torch.exp(-sxout)
                outputsx_np= cv2.resize(outputsx_np[0,0,:,:].cpu().detach().numpy(),(a,b))

                outputsy_np = torch.exp(-syout)
                outputsy_np= cv2.resize(outputsy_np[0,0,:,:].cpu().detach().numpy(),(a,b))

                outputlap_np = torch.exp(-lapout)
                outputlap_np= cv2.resize(outputlap_np[0,0,:,:].cpu().detach().numpy(),(a,b))


                clearsx_np = torch.exp(-true1_var)
                clearsx_np= cv2.resize(clearsx_np[0,0,:,:].cpu().detach().numpy(),(a,b))

                clearsy_np = torch.exp(-true2_var)
                clearsy_np= cv2.resize(clearsy_np[0,0,:,:].cpu().detach().numpy(),(a,b))

                clearlap_np = torch.exp(-true3_var)
                clearlap_np= cv2.resize(clearlap_np[0,0,:,:].cpu().detach().numpy(),(a,b))
                
          
                SSIM_val = SSIM.compute_ssim(np.clip(255*img_cc,0.0,255.0),np.clip(255*output_np,0.0,255.0),num_input_channels)
                PSNR_val = SSIM.cal_psnr(np.clip(255*img_cc,0.0,255.0),np.clip(255*output_np,0.0,255.0))     
                ssim+=SSIM_val;psnr+=PSNR_val
                s+=SSIM_val;p+=PSNR_val
                temp = np.concatenate((img_cc,np.exp(-noise_img), output_np, outputsx_np, clearsx_np, outputsy_np,  clearsy_np, outputlap_np,clearlap_np), axis=1)      
                if num_input_channels==3:
                    cv2.imwrite(result_syn + '/' + files[j][:-4] +'_%d_%d_Mix_%.4f_%.4f'%(epoch,noiselevel[i],SSIM_val,PSNR_val)+'.png',np.clip(temp[:,:,::-1]*255,0.0,255.0))
                else:
                    cv2.imwrite(result_syn + '/' + files[j][:-4] +'_%d_%d_Mix_%.4f_%.4f'%(noiselevel[i],epoch,SSIM_val,PSNR_val)+'.png',np.clip(temp*255,0.0,255.0))                       
 
        print('Synthetic Images Test: NoiseLevel is %d, SSIM is :%6f and PSNR is :%6f'%(noiselevel[i],ssim/len(files),psnr/len(files)))
        with open('./log/syntext.txt','a') as f:
            
            f.writelines('Synthetic Images Test: Epoch is %d,  NoiseLevel is %d, SSIM is :%6f and PSNR is :%6f'%(epoch,noiselevel[i],ssim/len(files),psnr/len(files)))
            f.writelines('\r\n')
    m.append(p/(len(files)*4))
    n.append(s/(len(files)*4))
    load_excel(m,n)
    return m,n
               
        
if __name__ == '__main__':
    checkpoint_dir = './checkpoint/'
    test_syn = './dataset/test'
    result_syn = './result/test'
 

    
    print('> Loading dataset ...')
    dataset_train_syn = Dataset(trainrgb=False,trainsyn = True, shuffle=True)
    loader_train_syn = DataLoader(dataset=dataset_train_syn, num_workers=0, batch_size=16, shuffle=True)
    
    #dataset_train_real = Dataset(trainrgb=False,trainsyn = False, shuffle=True)
    #loader_train_real = DataLoader(dataset=dataset_train_real, num_workers=0, batch_size=32, shuffle=True)
    
    num_input_channels = 1
    lr_update_freq = 40

    model, optimizer, cur_epoch = load_checkpoint(checkpoint_dir,num_input_channels)

    mix_loss = fixed_loss()
    msssim_loss = mix_loss.cuda()
    L1_loss = torch.nn.L1Loss(reduce=True, size_average=True)
    L1_loss = L1_loss.cuda()
    L2_loss = torch.nn.MSELoss(reduce=True, size_average=True)
    L2_loss = L2_loss.cuda()
    m = []; n =[]
    for epoch in range(cur_epoch, 200):
        #losses = AverageMeter()
        optimizer = adjust_learning_rate(optimizer, epoch, lr_update_freq)
        learnrate = optimizer.param_groups[-1]['lr']
        model.train()
        #train
        for i,data in enumerate(loader_train_syn,0):
            input_var , target_var , inputsx_var , targetsx_var , inputsy_var , targetsy_var , inputlap_var , targetlap_var = train_synthetic(data)   

            
            sxout,syout,lapout,x_out = model(input_var , inputsx_var , inputsy_var , inputlap_var)      

            #print(input_var.shape,outputvar_128.shape)                        
            loss = 0.5*L2_loss(x_out,target_var) + 0.2*L1_loss(x_out,target_var) + 0.1*L2_loss(sxout,targetsx_var)+\
                   0.1*L2_loss(syout,targetsy_var)+ 0.1*L2_loss(lapout,targetlap_var)

            
            #losses.update(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()   
            psnr_1 = train_psnr(target_var,x_out)

            
            print("[Synthetic_Epoch %d][%d/%d] lr :%f loss: %.4f PSNR_train: %.4f" %(epoch+1, i+1, len(loader_train_syn), learnrate, loss.item(), psnr_1))

       
        #test
        
        m ,n = test_synthetic(test_syn,result_syn, model,epoch,num_input_channels,m,n)  
        #test_real(test_c,test_n,result_real,model,epoch,num_input_channels)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict()}, is_best=0)
