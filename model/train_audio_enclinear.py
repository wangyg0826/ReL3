import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = "1"

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import l3_model
from esc50_loader import esc50_dataset
import pickle
import signal as os_signal
from datetime import datetime as dt
from pathlib import Path
import shutil
from util import stop_handler, STOP, audio_transform_raw
import numpy as np


def train_enclinear(clf, device, dataloader, val_dataloader, params, log_file):
    clf.to(device)
    optimizer = optim.Adam(clf.parameters(), lr=params.learning_rate, weight_decay = params.weight_decay)
    criterion = nn.CrossEntropyLoss().to(device)
    for epoch in range(int(params.epochs)):
        clf.train()
        loss_sum = 0.
        count = 0
        for idx, data in enumerate(dataloader):
            audio_batch, labels = data
            labels = labels.long().to(device)
            audio_batch = audio_batch.to(device)
            out = clf(audio_batch)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count += 1
            print("[{}]{}: loss = {}".format(dt.now().strftime("%Y-%m-%dT%H:%M:%S"),  count, loss.item()), file=log_file, flush=True)
            loss_sum += loss.item()
        if device == "cuda":
            torch.cuda.empty_cache()
        print("[{}]epoch {}, average loss = {}".format(dt.now().strftime("%Y-%m-%dT%H:%M:%S"), epoch+1, loss_sum / idx), flush=True, file=log_file)
        if (epoch+1) % 10 == 0:
            torch.save(clf.state_dict(), os.path.join(params.output_directory, "enclinear_{}.model".format(epoch+1)))
            correct = 0
            count = 0
            clf.eval()
            with torch.no_grad():
                for _, data in enumerate(val_dataloader):
                    audio_batch, labels = data
                    labels = labels.long().to(device)
                    audio_batch = audio_batch.to(device)
                    out = clf(audio_batch)
                    pred = out.argmax(dim=1)
                    correct += torch.sum(pred == labels).item()
                    count += labels.shape[0]
                if device == "cuda":
                    torch.cuda.empty_cache()
                print("[{}]epoch {}, validation accuracy = {}".format(dt.now().strftime("%Y-%m-%dT%H:%M:%S"), epoch+1, float(correct) / count), flush=True, file=log_file)
        if STOP.FLAG:
            torch.save(clf.state_dict(), os.path.join(params.output_directory, "enclinear_{}_stop.model".format(epoch+1)))
            print("[{}]epoch {}: forced stop".format(dt.now().strftime("%Y-%m-%dT%H:%M:%S"), epoch+1), file=log_file, flush=True)
            log_file.close()
            exit()
    torch.save(clf.state_dict(), os.path.join(params.output_directory, "enclinear_final.model"))
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HyperParameters')
    parser.add_argument("output_directory")
    parser.add_argument("-e", "--epochs", default = 50)
    parser.add_argument("-lr", "--learning_rate", type=float, default = 1e-3)
    parser.add_argument("--weight_decay", type=float, default = 1e-5)
    parser.add_argument("-b","--batch_size", type=int, default = 40)
    parser.add_argument("-vb","--validation_batch_size", type=int, default = 40)
    parser.add_argument("-v","--verbose", action="store_true", default=False)
    parser.add_argument("-ld", "--local_dir", default=r".\data\ESC-50")
    parser.add_argument("-fl", "--file_list", default=r".\data\ESC-50\files.txt")
    parser.add_argument("-enc", "--encoder", default=None)
    parser.add_argument("-model", "--pretrained_model", default=None)
    parser.add_argument("-sf","--shuffle", action="store_true", default=False)
    parser.add_argument("-t","--top", type=int, default=0)
    parser.add_argument("-cd", "--cache_dir", type=str, default=r".\data\ESC-50\cache_ae")
    parser.add_argument("-nw", "--num_workers", type=int, default=2)
    params = parser.parse_args()
    if not params.cache_dir:
        cache_dir = Path(params.local_dir, "cache_ae")
    else:
        cache_dir = Path(params.cache_dir)
    if not cache_dir.exists():
        cache_dir.mkdir()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_file = open(os.path.join(params.output_directory, "audio_ae_progress.log"), "a+")
    train_partitions = set([1, 2, 3, 4])
    val_partition = set([5])
    esc50_ds = esc50_dataset(params.local_dir, params.file_list, params.top, cache_dir, train_partitions, audio_transform_raw)
    n_class = len(esc50_ds.label_names)
    n_feature = 6*256
    val_esc50_ds = esc50_dataset(params.local_dir, params.file_list, params.top, cache_dir, val_partition, audio_transform_raw)
    dataloader = torch.utils.data.DataLoader(esc50_ds, batch_size=params.batch_size, drop_last=True, shuffle=params.shuffle, num_workers=params.num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_esc50_ds, batch_size=params.validation_batch_size, drop_last=True, shuffle=params.shuffle, num_workers=params.num_workers)
    if not params.pretrained_model:
        ae = l3_model.AudioAutoencoder()
        ae.load_state_dict(torch.load(params.encoder, map_location=torch.device(device)))
        clf = l3_model.AudioEncLinear(ae.encoder, n_feature, n_class)
        clf._initialize_weights()
    else:
        encoder_tmp = l3_model.AudioAutoencoder().encoder
        clf = l3_model.AudioEncLinear(encoder_tmp, n_feature, n_class)
        clf.load_state_dict(torch.load(params.pretrained_model, map_location=torch.device(device)))
    train_enclinear(clf, device, dataloader, val_dataloader, params, log_file)