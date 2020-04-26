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
from util import stop_handler, STOP, pick_false_audio_samples, audio_transform, audio_transform_mel
from test import test
from sklearn.linear_model import SGDClassifier
import numpy as np


def train_validation(audio_model, svm_clf, device, dataloader, val_dataloader, params, all_classes, log_file, leave_out):
    for epoch in range(int(params.epochs)):
        for _, data in enumerate(dataloader):
            audio_batch, labels = data
            labels = labels.numpy()
            audio_batch = audio_batch.to(device)
            feature = audio_model(audio_batch)
            feature = feature.cpu().numpy()
            svm_clf.partial_fit(feature, labels, all_classes)
        if device == "cuda":
            torch.cuda.empty_cache()
        print("[{}]epoch {} completed".format(dt.now().strftime("%Y-%m-%dT%H:%M:%S"), epoch+1), flush=True, file=log_file)
        if (epoch+1) % 10 == 0:
            with open(os.path.join(params.output_directory, "audio_svm_model_no_{}_{}.pkl".format(leave_out, epoch+1)), 'wb') as svm_file:
                pickle.dump(svm_clf, svm_file)    
            total = 0
            correct = 0
            for _, data in enumerate(val_dataloader):
                audio_batch, labels = data
                labels = labels.numpy()
                audio_batch = audio_batch.to(device)
                feature = audio_model(audio_batch)
                feature = feature.cpu().numpy()
                predict = svm_clf.predict(feature)
                total += labels.size
                correct += np.sum(np.equal(predict, labels))
            accuracy = float(correct) / total
            if device == "cuda":
                torch.cuda.empty_cache()
            print("[{}]epoch {}, validation accuracy = {}".format(dt.now().strftime("%Y-%m-%dT%H:%M:%S"), epoch+1, accuracy), flush=True, file=log_file)
        if STOP.FLAG:
            with open(os.path.join(params.output_directory, "audio_svm_model_no_{}_{}_stop.pkl".format(leave_out, epoch+1)), 'wb') as svm_file:
                pickle.dump(svm_clf, svm_file)
            print("[{}]epoch {}: forced stop".format(dt.now().strftime("%Y-%m-%dT%H:%M:%S"), epoch+1), file=log_file, flush=True)
            log_file.close()
            exit()
    return accuracy


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='HyperParameters')
    parser.add_argument("output_directory")
    parser.add_argument("-e", "--epochs", default = 50)
    #parser.add_argument("-lr", "--learning_rate", type=float, default = 1e-4)
    #parser.add_argument("--weight_decay", type=float, default = 1e-5)
    parser.add_argument("-b","--batch_size", type=int, default = 170)
    parser.add_argument("-vb","--validation_batch_size", type=int, default = 170)
    parser.add_argument("-v","--verbose", action="store_true", default=False)
    parser.add_argument("-ld", "--local_dir", default=r".\data\ESC-50")
    parser.add_argument("-fl", "--file_list", default=r".\data\ESC-50\files.txt")
    parser.add_argument("-model", "--pretrained_model", default=None)
    parser.add_argument("-sm", "--svm_models", default=None)
    parser.add_argument("-sf","--shuffle", action="store_true", default=False)
    parser.add_argument("-t","--top", type=int, default=0)
    parser.add_argument("-cd", "--cache_dir", type=str, default=r".\data\ESC-50\cache")
    parser.add_argument("-nw", "--num_workers", type=int, default=2)
    parser.add_argument("-nm","--no_mel", action="store_true", default=False)
    params = parser.parse_args()
    #create_partitions(params.local_dir)
    os_signal.signal(os_signal.SIGINT, stop_handler)
    os_signal.signal(os_signal.SIGTERM, stop_handler)
    if os.name != "nt":
        os_signal.signal(os_signal.SIGUSR1, stop_handler)
    if not params.cache_dir:
        cache_dir = Path(params.local_dir, "cache")
    else:
        cache_dir = Path(params.cache_dir)
    if not cache_dir.exists():
        cache_dir.mkdir()
    if params.no_mel:
        audio_transform_func = audio_transform
    else:
        audio_transform_func = audio_transform_mel
    model = l3_model.L3Net()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_file = open(os.path.join(params.output_directory, "audio_progress.log"), "a+")
    if params.pretrained_model is not None:
        model.load_state_dict(torch.load(params.pretrained_model,  map_location=torch.device(device)))
    else:
        model._initialize_weights()
    audio_model = l3_model.AudioFeature(model.audiosub)
    if device != "cuda":
        print("[{}]no cuda device".format(dt.now().strftime("%Y-%m-%dT%H:%M:%S")), flush=True, file=log_file)
    audio_model.audioSubNet = audio_model.audioSubNet.to(device)
    audio_model = audio_model.to(device)
    svm_models = []
    if params.svm_models is not None:
        svm_models = params.svm_models.split(',')
    all_partitions = set([1, 2, 3, 4, 5])
    accuracy_array = np.empty(5, dtype=np.float32)
    for i in range(1, 6):
        leaveout_partition = set([i])
        include_partitions = all_partitions - leaveout_partition
        esc50_ds = esc50_loader.esc50_dataset(params.local_dir, params.file_list, params.top, cache_dir, include_partitions, audio_transform_func)
        #print(len(esc50_ds.file_list))
        all_classes = [v for k, v  in esc50_ds.label_names.items()]
        val_esc50_ds = esc50_loader.esc50_dataset(params.local_dir, params.file_list, params.top, params.cache_dir, leaveout_partition, audio_transform_func)
        #print(len(val_esc50_ds.file_list))
        dataloader = torch.utils.data.DataLoader(esc50_ds, batch_size=params.batch_size, drop_last=True, shuffle=params.shuffle, num_workers=params.num_workers)
        val_dataloader = torch.utils.data.DataLoader(val_esc50_ds, batch_size=params.validation_batch_size, drop_last=True, shuffle=params.shuffle, num_workers=params.num_workers)
        if svm_models:
            with open(svm_models[i-1], "rb") as svm_model_file:
                svm_clf = pickle.load(svm_model_file)
        else:
            svm_clf = SGDClassifier(loss="hinge", penalty="l1", max_iter=1, n_jobs=-1)
        accuracy = train_validation(audio_model, svm_clf, device, dataloader, val_dataloader, params, all_classes, log_file, i)
        print("[{}]leave out {}, accuracy = {}".format(dt.now().strftime("%Y-%m-%dT%H:%M:%S"), i, accuracy), flush=True, file=log_file)
        accuracy_array[i-1] = accuracy
    accuracy = np.mean(accuracy_array)
    print("[{}]average accuracy = {}".format(dt.now().strftime("%Y-%m-%dT%H:%M:%S"), accuracy), flush=True, file=log_file)
    log_file.close()
