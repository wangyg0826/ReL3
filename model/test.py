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
import l3_model
import flickr_loader
import signal as os_signal
from datetime import datetime as dt
from pathlib import Path
import shutil
from util import stop_handler, STOP, pick_false_audio_samples, audio_transform_mel, audio_transform


def test(model, dataloader, params, criterion = nn.CrossEntropyLoss()):
    os_signal.signal(os_signal.SIGINT, stop_handler)
    os_signal.signal(os_signal.SIGTERM, stop_handler)
    if os.name != "nt":
        os_signal.signal(os_signal.SIGUSR1, stop_handler)
    with torch.no_grad():
        model.eval()
        log_file = open(os.path.join(params.output_directory, "progress.log"), "a+")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("[{}]test start".format(dt.now().strftime("%Y-%m-%dT%H:%M:%S")), file=log_file)
        criterion = criterion.to(device)
        model = model.to(device)
        true_positive = 0
        false_negative = 0
        true_negative = 0
        false_positive = 0
        count = 0
        for _, data in enumerate(dataloader):
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
            correct = (pred_label == label_batch.data)
            #print(correct)
            cur_true_positive = torch.sum(correct[0:cur_batch_size]).item()
            #print(cur_true_positive)
            true_positive += cur_true_positive
            false_negative += cur_batch_size - cur_true_positive
            cur_true_negative = torch.sum(correct[cur_batch_size:2*cur_batch_size]).item()
            true_negative += cur_true_negative
            false_positive += cur_batch_size - cur_true_negative
            # loss = criterion(pred, label_batch)
            # if params.verbose:
            #     print("[{}]{}: loss = {}".format(dt.now().strftime("%Y-%m-%dT%H:%M:%S"),  count, loss.data), file=log_file, flush=True)
            if STOP.FLAG:
                print("[{}]{}: forced stop".format(dt.now().strftime("%Y-%m-%dT%H:%M:%S"), count), file=log_file, flush=True)
                log_file.close()
                exit()
        #print(true_positive, false_negative, true_negative, false_positive)
        if device == 'cuda':
            torch.cuda.empty_cache()
        total = true_positive + false_negative + true_negative + false_positive
        accuracy = float(true_positive + true_negative) / total
        true_positive_rate = float(true_positive) / (true_positive +  false_negative)
        false_negative_rate = float(false_negative) / (true_positive +  false_negative)
        true_negative_rate = float(true_negative) / (true_negative +  false_positive)
        false_positive_rate = float(false_positive) / (true_negative +  false_positive)
        print("[{}]true positive rate = {}; false negative rate = {}; true negative rate = {}; false positive rate = {}".format(dt.now().strftime("%Y-%m-%dT%H:%M:%S"), true_positive_rate, false_negative_rate, true_negative_rate, false_positive_rate), file=log_file)
        print("[{}]total = {}; accuracy = {}".format(dt.now().strftime("%Y-%m-%dT%H:%M:%S"), total, accuracy), file=log_file)
        log_file.close()
        return model


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='HyperParameters')
    parser.add_argument("output_directory")
    #parser.add_argument("-e", "--epochs", default = 50)
    #parser.add_argument("-lr", "--learning_rate", type=float, default = 1e-4)
    #parser.add_argument("--weight_decay", type=float, default = 1e-5)
    parser.add_argument("-b","--batch_size", type=int, default = 4)
    parser.add_argument("-v","--verbose", action="store_true", default=False)
    parser.add_argument("-ld", "--local_dir", default="")
    parser.add_argument("-rd", "--remote_dir", default="")
    parser.add_argument("-fl", "--file_list", default="")
    parser.add_argument("-s", "--server", default="ftp.box.com")
    parser.add_argument("-u", "--username", default="")
    parser.add_argument("-st", "--secret", default="")
    parser.add_argument("-model", "--pretrained_model", default=None)
    parser.add_argument("-sf","--shuffle", action="store_true", default=False)
    parser.add_argument("-t","--top", type=int, default=0)
    #parser.add_argument("-ss","--save_step", type=int, default=100)
    parser.add_argument("-cd", "--cache_dir", type=str, default="")
    parser.add_argument("-nw", "--num_workers", type=int, default=1)
    parser.add_argument("-nm","--no_mel", action="store_true", default=False)
    params = parser.parse_args()
    if not params.cache_dir:
        cache_dir = Path(params.local_dir, "cache_test")
    else:
        cache_dir = params.cache_dir
    if not cache_dir.exists():
        cache_dir.mkdir()
    if params.no_mel:
        audio_transform_func = audio_transform
    else:
        audio_transform_func = audio_transform_mel
    flickr_ds = flickr_loader.flickr_dataset(params.local_dir, params.file_list, params.remote_dir, params.server, params.username, params.secret, params.top, cache_dir, audio_transform_func)
    dataloader = torch.utils.data.DataLoader(flickr_ds, batch_size=params.batch_size, drop_last=True, shuffle=params.shuffle, num_workers=params.num_workers)
    model = l3_model.L3Net()
    if params.pretrained_model is not None:
        model.load_state_dict(torch.load(params.pretrained_model))
    else:
        model._initialize_weights()
    model = test(model, dataloader, params)
