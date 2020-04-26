import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = "1"

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import l3_model
import esc50_loader
import pickle
import signal as os_signal
from datetime import datetime as dt
from pathlib import Path
import shutil
from util import stop_handler, STOP, pick_false_audio_samples
from test import test
from sklearn.linear_model import SGDClassifier
import numpy as np
import torchvision


def train_validation(model, cls, train_dataloader, test_dataloader, device, params, log_file):    
    optimizer = optim.Adam(cls.parameters(), lr=params.learning_rate, weight_decay = params.weight_decay)
    criterion = nn.CrossEntropyLoss().to(device)
    
    count=0
    mp = nn.MaxPool2d(kernel_size=3, stride=3).to(device)
    for epoch in range(int(params.epochs)):
        for _, data in enumerate(train_dataloader):
            count+=1
            # print(data)
            visual_batch, labels = data
            visual_batch = visual_batch.to(device)
            labels = labels.long().to(device)
            with torch.no_grad():
                pred = model.visualsub.visual_feature(visual_batch)
                pred = mp(pred).view(pred.shape[0], -1)
            pred = cls(pred)
            print(pred.shape, labels.shape)
            loss = criterion(pred, labels)
            
            pred_data = pred.data
            pred_label = torch.argmax(pred_data, dim=1)
            train_accuracy = torch.mean((pred_label == labels.data).float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if params.verbose:
                print("[{}]{}: train_accuracy = {}".format(dt.now().strftime("%Y-%m-%dT%H:%M:%S"), count, train_accuracy), file=log_file)
                print("[{}]{}: loss = {}".format(dt.now().strftime("%Y-%m-%dT%H:%M:%S"),  count, loss.data), file=log_file, flush=True)
        if device == "cuda":
            torch.cuda.empty_cache()
        print("[{}]epoch {} completed".format(dt.now().strftime("%Y-%m-%dT%H:%M:%S"), epoch+1), flush=True, file=log_file)
        total = 0
        correct = 0
        for _, data in enumerate(test_dataloader):
            visual_batch, labels = data
            visual_batch = visual_batch.to(device)
            labels = labels.long().to(device)
            with torch.no_grad():
                pred = model.visualsub.visual_feature(visual_batch)
                pred = mp(pred).view(pred.shape[0], -1)
                pred = cls(pred)
                # loss = criterion(pred, labels)
            
            pred_data = pred.data
            pred_label = torch.argmax(pred_data, dim=1)
            correct += torch.sum((pred_label == labels.data).float())
            total += visual_batch.shape[0]
            
        accuracy = float(correct) / total
        if device == "cuda":
            torch.cuda.empty_cache()
        print("[{}]epoch {}, validation accuracy = {}".format(dt.now().strftime("%Y-%m-%dT%H:%M:%S"), epoch+1, accuracy), flush=True, file=log_file)
        if STOP.FLAG:
            print("[{}]epoch {}: forced stop".format(dt.now().strftime("%Y-%m-%dT%H:%M:%S"), epoch+1), file=log_file, flush=True)
            log_file.close()
            exit()
    return accuracy


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='HyperParameters')
    parser.add_argument("output_directory")
    parser.add_argument("-ld", "--local_dir")
    parser.add_argument("-e", "--epochs", default = 50)
    parser.add_argument("-lr", "--learning_rate", type=float, default = 1e-4)
    parser.add_argument("--weight_decay", type=float, default = 5e-7)
    parser.add_argument("-b","--batch_size", type=int, default = 100)
    parser.add_argument("-v","--verbose", action="store_true", default=False)
    parser.add_argument("-model", "--pretrained_model", default=None)
    parser.add_argument("-sf","--shuffle", action="store_true", default=False)
    parser.add_argument("-nw", "--num_workers", type=int, default=2)
    params = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_file = open(os.path.join(params.output_directory, "visual_progress.log"), "a+")
    if device != "cuda":
        print("[{}]no cuda device".format(dt.now().strftime("%Y-%m-%dT%H:%M:%S")), flush=True, file=log_file)
    
    
    transfrom = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    
    dataset_tr = torchvision.datasets.STL10(params.local_dir+'/data', 'train', download=False, transform=transfrom)
    dataset_te = torchvision.datasets.STL10(params.local_dir+'/data', 'test', download=False, transform=transfrom)
    dataloader_tr = torch.utils.data.DataLoader(dataset_tr, batch_size=params.batch_size, shuffle=params.shuffle)
    dataloader_te = torch.utils.data.DataLoader(dataset_te, batch_size=params.batch_size, shuffle=params.shuffle)
    
    model = l3_model.L3Net()
    if params.pretrained_model is not None:
        model.load_state_dict(torch.load(params.pretrained_model, map_location=torch.device(device)))
    else:
        model._initialize_weights()
        
    cls = nn.Sequential(nn.Linear(8192, 10)).to(device)
    model = model.to(device)
    train_validation(model, cls, dataloader_tr, dataloader_te, device, params, log_file)     
    log_file.close()
