'''
ECOLÉ source code.
ECOLÉ is a deep learning based WES CNV caller tool.
This script, ECOLÉ_call.py, is only used to load the weights of pre-trained models
and use them to perform CNV calls.
'''

import os
import torch
import math
torch.cuda.empty_cache()
from tqdm import tqdm
import sys
from performer_pytorch import Performer
import numpy as np
from scipy.stats import mode
from einops import rearrange, repeat
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.keras.preprocessing import sequence
from torch.nn.parameter import Parameter
from torch.utils.data import TensorDataset, DataLoader,Dataset
from torchvision.datasets import DatasetFolder
import argparse
import datetime
from net_my_gm_v2 import CNVcaller

'''
Helper function to print informative messages to the user.
'''
def message(message):
    print("[",datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"]\t", message)

'''
Helper function to calculate metrics.
'''
def calculate_metrics(tp_nocall, tp_plus_fp_nocall, tp_duplication, tp_plus_fp_duplication, tp_deletion, tp_plus_fp_deletion, tp_plus_fn_nocall, tp_plus_fn_duplication, tp_plus_fn_deletion):
    nocall_prec = tp_nocall / (tp_plus_fp_nocall+1e-15) 
    dup_prec = tp_duplication / (tp_plus_fp_duplication+1e-15) 
    del_prec = tp_deletion / (tp_plus_fp_deletion+1e-15) 
    nocall_recall = tp_nocall / (tp_plus_fn_nocall+1e-15) 
    dup_recall = tp_duplication / (tp_plus_fn_duplication+1e-15) 
    del_recall = tp_deletion / (tp_plus_fn_deletion+1e-15) 

    return nocall_prec, dup_prec, del_prec, nocall_recall, dup_recall, del_recall



''' 
Perform I/O operations.
'''

description = "ECOLÉ is a deep learning based WES CNV caller tool. \
            For academic research the software is completely free, although for \
            commercial usage the software is licensed. \n \
            please see ciceklab.cs.bilkent.edu.tr/ECOLÉlicenceblablabla."

parser = argparse.ArgumentParser(description=description)

'''
Required arguments group:
(i) -bs, batch size to be used in the training. 
(ii) -i, input dataset path comprised of WES samples with read depth data.
(iii) -o, relative or direct output directory path to save ECOLÉ output model weights.
(iv) -n, The path for mean&std stats of read depth values.
'''

required_args = parser.add_argument_group('Required Arguments')
opt_args = parser.add_argument_group("Optional Arguments")

required_args.add_argument("-bs", "--batch_size", help="Batch size to be used in the training.", required=True)

required_args.add_argument("-i", "--input", help="Relative or direct path to input dataset for ECOLÉ model training, these are the processed samples for training.", required=True)

required_args.add_argument("-o", "--output", help="Relative or direct output directory path to write ECOLÉ output model weights.", required=True)

required_args.add_argument("-n", "--normalize", help="Please provide the path for mean&std stats of read depth values to normalize. \n \
                                                    These values are obtained precalculated from the training dataset before the pretraining.", required=True)

required_args.add_argument("-e", "--epochs", help="Please provide the number of epochs the training will be performed.", required=True)

required_args.add_argument("-lr", "--learning_rate", help="Please provide the learning rate to be used in training.", required=True)
required_args.add_argument("-nl", "--n_layer", help="Please provide the learning rate to be used in training.", required=True)
required_args.add_argument("-w", "--weight", help="Please provide the learning rate to be used in training.", required=True)

opt_args.add_argument("-m", "--model", help="Model path if you want to finetune existed model")
opt_args.add_argument("-g", "--gpu", help="Specify gpu", required=False)


'''
Optional arguments group:
-v or --version, version check
-h or --help, help
-g or --gpu, specify gpu
-
'''

parser.add_argument("-V", "--version", help="show program version", action="store_true")
args = parser.parse_args()

if args.version:
    print("ECOLÉ version 0.1")

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

if args.gpu:
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
    message("Using GPU!")
else:
    message("Using CPU!")

os.makedirs(os.path.join('ckpt',args.output), exist_ok=True)
os.makedirs(os.path.join('model',args.output), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

bs = int(args.batch_size)
N_EPOCHS = int(args.epochs)
weight = float(args.weight)
LR = float(args.learning_rate)


EXON_SIZE = 1000
PATCH_SIZE = 1
NO_LAYERS = int(args.n_layer)
FEATURE_SIZE = 192
NUM_CLASS = 3

class MyDataset(DatasetFolder):
    def _find_classes(self, directory: str):
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {"nocall":0,"deletion":2,"duplication":1}
      
        return classes, class_to_idx

print("input: ", args.input)
my_dataset = MyDataset(args.input, loader=np.load, extensions = (".npy",".npz"))
dataset_length = len(my_dataset)
print("Dataset Length:", dataset_length)

file1 = open(args.normalize, 'r')
line = file1.readline()
means_ = float(line.split(",")[0])
stds_ = float(line.split(",")[1])
sub_train_ = DataLoader(my_dataset, batch_size = bs, shuffle=True, pin_memory=True, num_workers=8) # create your dataloader

model = CNVcaller(EXON_SIZE, PATCH_SIZE, NO_LAYERS, FEATURE_SIZE, NUM_CLASS).to(device)
if args.model:
    model.load_state_dict(torch.load(args.model))

print("number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))


from torch.utils.data import DataLoader

def train_func(epoch,trainloader):

    data = trainloader
    
    train_loss, train_acc = 0, 0

    tp_tn_fp_fn = 0
    tp_nocall = 0
    tp_duplication = 0
    tp_plus_fp_nocall = 0
    tp_plus_fp_duplication = 0 
    tp_plus_fp_deletion = 0

    tp_plus_fn_nocall = 0
    tp_plus_fn_duplication = 0
    tp_plus_fn_deletion = 0
    tp_deletion = 0

    model.train()
    for i, (exons, labels) in enumerate(data):
        ind = torch.arange(labels.size(0))
        optimizer.zero_grad()
        
        labels = labels.long()
        exons, labels = exons.to(device),  labels.to(device)
        
        exons = exons.float()
        #mask = torch.logical_not(exons == -1)
        mask = torch.logical_and(exons != -1, exons != 0)

        mask = torch.squeeze(mask)
        real_mask = torch.ones(labels.size(0),2005, dtype=torch.bool).to(device)
        
        real_mask[:,1:] = mask
       
        output1 = model(exons, real_mask)       
     
        loss1 = CEloss(output1, labels) 

  
        train_loss += loss1.item()
        loss1.backward()
    
       
        optimizer.step()
        
        
        _, predicted = torch.max(output1.data, 1)
   
     
        tp_nocall += (torch.logical_and(predicted == labels,predicted == 0)).sum().item() 
        tp_duplication += (torch.logical_and(predicted == labels,predicted == 1)).sum().item() 
        tp_deletion += (torch.logical_and(predicted == labels,predicted == 2)).sum().item()

        train_acc += (predicted == labels).sum().item()
        tp_plus_fp_nocall += (predicted == 0).sum().item()
        tp_plus_fp_duplication += (predicted == 1).sum().item()
        tp_plus_fp_deletion += (predicted == 2).sum().item()
   
        
        tp_plus_fn_nocall += (labels == 0).sum().item()
        tp_plus_fn_duplication += (labels == 1).sum().item()
        tp_plus_fn_deletion += (labels == 2).sum().item()
        tp_tn_fp_fn += labels.size(0)

        if i % 5000 == 0 and i > 0:
            nocall_prec, dup_prec, del_prec, nocall_recall, dup_recall, del_recall = calculate_metrics(tp_nocall, tp_plus_fp_nocall, tp_duplication, tp_plus_fp_duplication, tp_deletion, tp_plus_fp_deletion, tp_plus_fn_nocall, tp_plus_fn_duplication, tp_plus_fn_deletion)    
            #message(f"Model weights are saved for Epoch: {epoch}")
            #torch.save(model.state_dict(), os.path.join("ckpt", args.output, f"ecole_depth{str(NO_LAYERS)}_lr{str(LR)}_epoch{epoch}_batch{i}.pt"))
            message(f'Epoch {int(epoch)+1}: Batch no: {i}\tLoss: {train_loss / (i+1):.4f}(train)\t|\tNocall_prec: {nocall_prec * 100:.1f}%(train)|\tDup_prec: {dup_prec * 100:.1f}%(train)|\tDel_prec: {del_prec * 100:.1f}%(train)|\tNocall_recall: {nocall_recall * 100:.1f}%(train)|\tDup_recall: {dup_recall * 100:.1f}%(train)|\tDel_recall: {del_recall * 100:.1f}%(train)')
    
    nocall_prec, dup_prec, del_prec, nocall_recall, dup_recall, del_recall = calculate_metrics(tp_nocall, tp_plus_fp_nocall, tp_duplication, tp_plus_fp_duplication, tp_deletion, tp_plus_fp_deletion, tp_plus_fn_nocall, tp_plus_fn_duplication, tp_plus_fn_deletion)
    
    return train_loss / len(data), ( train_acc / tp_tn_fp_fn), nocall_prec,dup_prec,del_prec, nocall_recall, dup_recall,del_recall


import time
from torch.utils.data.dataset import random_split

print('learning rate: ',LR)
print('epochs : ',N_EPOCHS)
print('batch size : ',bs)

min_valid_loss = float('inf')

optimizer = torch.optim.Adam(model.parameters(),lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,N_EPOCHS)

weight_class0, weight_class1, weight_class2 = 1.0, weight, weight
class_weights = [weight_class0, weight_class1, weight_class2]
CEloss = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))
print("weight: ", weight_class0, weight_class1, weight_class2)
#CEloss = nn.CrossEntropyLoss()

message("Starting training...")


for epoch in range(N_EPOCHS):
    
    start_time = time.time()

    train_loss, train_acc,nocall_prec,dup_prec,del_prec, nocall_recall, dup_recall,del_recall = train_func(epoch, sub_train_)
    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60
    message(f"Model weights are saved for Epoch: {epoch}")
    torch.save(model.state_dict(), os.path.join("ckpt", args.output, f"ecole_depth{str(NO_LAYERS)}_lr{str(LR)}_epoch{epoch}.pt"))
    scheduler.step()
    message('Epoch: %d | time in %d minutes, %d seconds' % (epoch + 1, mins, secs))
    #message('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
    message(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)|\tNocall_prec: {nocall_prec * 100:.1f}%(train)|\tDup_prec: {dup_prec * 100:.1f}%(train)|\tDel_prec: {del_prec * 100:.1f}%(train)|\tNocall_recall: {nocall_recall * 100:.1f}%(train)|\tDup_recall: {dup_recall * 100:.1f}%(train)|\tDel_recall: {del_recall * 100:.1f}%(train)')
torch.save(model.state_dict(), os.path.join('model',args.output, f"ecole_depth{str(NO_LAYERS)}_lr{str(LR)}.pt"))
   
