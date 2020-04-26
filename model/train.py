# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 20:19:34 2020

@author: z7125
"""
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = "1"

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import l3_model
import flickr_loader
import pickle
import signal as os_signal
from datetime import datetime as dt
from pathlib import Path
import shutil
from util import stop_handler, STOP, pick_false_audio_samples, audio_transform, audio_transform_mel
from test import test

def train(model, dataloader, params, val_dataloader, criterion = nn.CrossEntropyLoss()):
    os_signal.signal(os_signal.SIGINT, stop_handler)
    os_signal.signal(os_signal.SIGTERM, stop_handler)
    if os.name != "nt":
        os_signal.signal(os_signal.SIGUSR1, stop_handler)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay = params.weight_decay)
    loss_hist = []
    log_file = open(os.path.join(params.output_directory, "progress.log"), "a+")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != "cuda":
        print("[{}]no cuda device, quit".format(dt.now().strftime("%Y-%m-%dT%H:%M:%S")), file=log_file, flush=True)
        log_file.close()
        exit()
    print("[{}]train start".format(dt.now().strftime("%Y-%m-%dT%H:%M:%S")), file=log_file)
    criterion = criterion.to(device)
    model = model.to(device)
    count = 0
    for epoch in range(int(params.epochs)):
        model.train()
        for idx, data in enumerate(dataloader):
            count += 1
            image_batch, true_audio_batch, false_audio_batch = data
            cur_batch_size = image_batch.shape[0]
            image_batch = image_batch.to(device)
            image_batch = torch.cat([image_batch, image_batch], dim=0)
            true_audio_batch = true_audio_batch.to(device)
            false_audio_batch = false_audio_batch.to(device)
            #false_audio_batch = pick_false_audio_samples(true_audio_batch)
            audio_batch = torch.cat([true_audio_batch, false_audio_batch], dim=0)
            pred = model(image_batch, audio_batch)
            pred_data = pred.data
            pred_label = torch.argmax(pred_data, dim=1)
            #print(pred_data)
            label_batch = torch.cat([torch.ones(cur_batch_size), torch.zeros(cur_batch_size)]).long().to(device)
            train_accuracy = torch.mean((pred_label == label_batch.data).float())
            loss = criterion(pred, label_batch)
            loss_hist.append(loss.item())
            
            if params.verbose:
                print("[{}]{}: train_accuracy = {}".format(dt.now().strftime("%Y-%m-%dT%H:%M:%S"), count, train_accuracy), file=log_file)
                print("[{}]{}: loss = {}".format(dt.now().strftime("%Y-%m-%dT%H:%M:%S"),  count, loss.data), file=log_file, flush=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if STOP.FLAG:
                torch.save(model.state_dict(), os.path.join(params.output_directory, "l3_model_{}_stop.model".format(count)))
                with open(os.path.join(params.output_directory, "loss_hist.pkl"), 'wb') as loss_hist_file:
                    pickle.dump(loss_hist, loss_hist_file)
                print("[{}]{}: forced stop".format(dt.now().strftime("%Y-%m-%dT%H:%M:%S"), count), file=log_file, flush=True)
                log_file.close()
                exit()
            if count % params.save_step == 0:
                torch.save(model.state_dict(), os.path.join(params.output_directory, "l3_model_{}.model".format(count)))
                with open(os.path.join(params.output_directory, "loss_hist.pkl"), 'wb') as loss_hist_file:
                    pickle.dump(loss_hist, loss_hist_file)
        if device == "cuda":
            torch.cuda.empty_cache()
        if params.verbose:
            print("[{}]epoch {} completed".format(dt.now().strftime("%Y-%m-%dT%H:%M:%S"), epoch+1), file=log_file, flush=True)
            test(model, val_dataloader, params)
    log_file.close()
    return model

def create_partitions(local_dir):
    for i in range(97):
        partition_path = Path(local_dir, str(i))
        if not partition_path.exists():
            try:
                partition_path.mkdir()
            except:
                pass


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='HyperParameters')
    parser.add_argument("output_directory")
    parser.add_argument("-e", "--epochs", default = 50)
    parser.add_argument("-lr", "--learning_rate", type=float, default = 1e-4)
    parser.add_argument("--weight_decay", type=float, default = 1e-5)
    parser.add_argument("-b","--batch_size", type=int, default = 4)
    parser.add_argument("-v","--verbose", action="store_true", default=False)
    parser.add_argument("-ld", "--local_dir", default="")
    parser.add_argument("-rd", "--remote_dir", default="")
    parser.add_argument("-fl", "--file_list", default="")
    parser.add_argument("-val", "--val_file_list", default="")
    parser.add_argument("-s", "--server", default="")
    parser.add_argument("-u", "--username", default="")
    parser.add_argument("-st", "--secret", default="")
    parser.add_argument("-model", "--pretrained_model", default=None)
    parser.add_argument("-sf","--shuffle", action="store_true", default=False)
    parser.add_argument("-t","--top", type=int, default=0)
    parser.add_argument("-ss","--save_step", type=int, default=1000)
    parser.add_argument("-cd", "--cache_dir", type=str, default="")
    parser.add_argument("-vcd", "--val_cache_dir", type=str, default="")
    parser.add_argument("-nw", "--num_workers", type=int, default=4)
    parser.add_argument("-nm","--no_mel", action="store_true", default=False)
    params = parser.parse_args()
    #create_partitions(params.local_dir)
    if not params.cache_dir:
        cache_dir = Path(params.local_dir, "cache")
    else:
        cache_dir = Path(params.cache_dir)
    if not cache_dir.exists():
        cache_dir.mkdir()
    if not params.val_cache_dir:
        val_cache_dir = Path(params.local_dir, "val_cache")
    else:
        val_cache_dir = Path(params.val_cache_dir)
    if not val_cache_dir.exists():
        val_cache_dir.mkdir()
    # val_top = max(round(params.top / 20), 10)
    if params.no_mel:
        audio_transform_func = audio_transform
    else:
        audio_transform_func = audio_transform_mel
    flickr_ds = flickr_loader.flickr_dataset(params.local_dir, params.file_list, params.remote_dir, params.server, params.username, params.secret, params.top, cache_dir, audio_transform_func)
    val_flickr_ds = flickr_loader.flickr_dataset(params.local_dir, params.val_file_list, params.remote_dir, params.server, params.username, params.secret, 0, val_cache_dir, audio_transform_func)
    dataloader = torch.utils.data.DataLoader(flickr_ds, batch_size=params.batch_size, drop_last=True, shuffle=params.shuffle, num_workers=params.num_workers)
    val_dataloader = None
    val_dataloader = torch.utils.data.DataLoader(val_flickr_ds, batch_size=params.batch_size, drop_last=True, shuffle=params.shuffle, num_workers=params.num_workers)
    model = l3_model.L3Net()
    if params.pretrained_model is not None:
        model.load_state_dict(torch.load(params.pretrained_model))
    else:
        model._initialize_weights()
    model = train(model, dataloader, params, val_dataloader)
    torch.save(model.state_dict(), os.path.join(params.output_directory, 'l3_model_final.model'))
    
    
    
    