from torch.utils.data import Dataset
import torchvision.io
import ftplib
from pathlib import Path
import time
import torch
import numpy as np
import torchvision.transforms
import os
import random
from scipy import signal
import sys
import tempfile
import shutil
import threading
import string
import pickle
import signal as os_signal
from util import stop_handler, get_partition
import math
import av

def get_connection(server, username, password):
    conn = ftplib.FTP_TLS(server, timeout=30.0)
    conn.login(username, password)
    return conn



def random_string(length):
    ret = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(length)])
    return ret

class flickr_dataset(Dataset):
    def read_audio_and_transform(self, filename, position):
        container = av.open(filename)
        audio_stream = container.streams.audio[0]
        afps = round(float(1/audio_stream.time_base))
        #print(afps)
        buffer = []
        for audio_frame in container.decode(audio_stream):
            buffer.append(audio_frame.to_ndarray())
        #print(buffer[0].dtype)
        audio = np.hstack(buffer)
        #print(audio.dtype)
        total_audio_frames = audio.shape[1]
        #print(total_audio_frames)
        if total_audio_frames < afps:
            afps = total_audio_frames
        audio_start_idx = math.floor(position * (total_audio_frames - afps))
        audio_end_idx = min(audio_start_idx + afps, total_audio_frames)
        audio = audio[:, audio_start_idx:audio_end_idx]
        audio = np.mean(audio, axis=0, keepdims=False)
        #print(np.min(audio), np.max(audio))
        audio = self.audio_transform_func(audio)
        #print(audio.dtype)
        return audio

    def __init__(self, local_dir, file_list_filename, remote_dir, server, username, secret_filename, top, cache_dir, audio_transform_func):
        #self.transform = None
        os_signal.signal(os_signal.SIGINT, stop_handler)
        os_signal.signal(os_signal.SIGTERM, stop_handler)
        if os.name != "nt":
            os_signal.signal(os_signal.SIGUSR1, stop_handler)
        secret_file = open(secret_filename, "r")
        password = secret_file.readline().strip()
        self.server = server
        self.username = username
        self.password = password
        self.remote_dir = remote_dir
        self.audio_transform_func = audio_transform_func
        #self.conn = get_connection(self.server, self.username, self.password)
        self.file_list = []
        count = 0
        with Path(file_list_filename).open('r') as file_list_file:
            for line in file_list_file:
                line = line.strip()
                if line:
                    filename, timepoint, false_filename, false_timepoint = line.split(';')
                    timepoint = float(timepoint)
                    false_timepoint = float(timepoint)
                    if top != 0:
                        if count < top:
                            self.file_list.append((filename, timepoint, false_filename, false_timepoint))
                            count += 1
                        else:
                            break
                    else:
                        self.file_list.append((filename, timepoint, false_filename, false_timepoint))

        self.local_dir = local_dir
        self.cache_dir = cache_dir
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                          torchvision.transforms.Resize(256),
                                                          torchvision.transforms.RandomCrop((224, 224)),
                                                          torchvision.transforms.ColorJitter(brightness=0.5, saturation=0.5),
                                                          torchvision.transforms.RandomHorizontalFlip(),
                                                          torchvision.transforms.ToTensor()])

    def __len__(self):
        return len(self.file_list)

    def download_from_ftp(self, partition, path_wo_filename, filename, full_path_str):
        while True:
            tmp_filename_str = "{}_{}.{}".format(filename.stem, random_string(10), filename.suffix)
            tmp_file_path = Path(path_wo_filename, tmp_filename_str)
            try:
                tmp_file_path.touch()
                break
            except:
                pass
        file_ = tmp_file_path.open("wb")
        ftp_path = Path(self.remote_dir, str(partition), filename)
        conn = get_connection(self.server, self.username, self.password)
        max_retry = 5
        retry = 0
        while retry < max_retry:
            if retry > 0:
                time.sleep(0.5)
            try:
                conn.retrbinary("RETR {}".format(ftp_path.as_posix()), file_.write)
                break
            except Exception as e:
                print(e, file=sys.stderr)
                retry += 1
        file_.close()
        if retry == max_retry:
            if tmp_file_path.exists():
                os.remove(tmp_file_path)
            raise InvalidVideo()
        else:
            try:
                if not os.path.exists(full_path_str):
                    shutil.move(tmp_file_path, full_path_str)
                return full_path_str
            except:
                if os.name == "nt":
                    return str(tmp_file_path)
                else:
                    return tmp_file_path.as_posix()

    def read_video(self, filename, position):
        filename = Path(filename)
        partition = get_partition(filename)
        path_wo_filename = Path(self.local_dir, str(partition))
        if not path_wo_filename.exists():
            try:
                os.mkdir(path_wo_filename)
            except:
                pass
        full_path = Path(self.local_dir, str(partition), filename)
        if os.name == "nt":
            full_path_str = str(full_path)
        else:
            full_path_str = full_path.as_posix()
        if not full_path.exists():
            full_path_str = self.download_from_ftp(partition, path_wo_filename, filename, full_path_str)
        pts, vfps = torchvision.io.read_video_timestamps(full_path_str, pts_unit="sec")
        vfps = round(vfps)
        total_video_frames = len(pts)
        if total_video_frames < vfps:
            vfps = total_video_frames
        video_start_idx = math.floor(position * (total_video_frames - vfps))
        video_end_idx_minus_1 = video_start_idx + vfps - 1
        video, _, _ = torchvision.io.read_video(full_path_str, pts[video_start_idx], pts[video_end_idx_minus_1], pts_unit="sec")
        audio = self.read_audio_and_transform(full_path_str, position)
        return video, audio

    def __getitem__(self, idx):
        #print(self.file_list[idx])
        read_from_path = Path(self.cache_dir, "{}.pkl".format(idx))
        if read_from_path is not None and read_from_path.exists():
            try:
                with read_from_path.open("rb") as read_from:
                    file_list_line, frame_buffer, audio, false_audio = pickle.load(read_from)
                    if file_list_line == self.file_list[idx]:
                        #print("load from cached file")
                        frame_idx = random.randint(0, len(frame_buffer)-1)
                        frame = frame_buffer[frame_idx]
                        return frame, audio, false_audio
                read_from_path.unlink()   
            except Exception as e:
                print("load from cache error", file=sys.stderr)
                print(self.file_list[idx], e, file=sys.stderr)
        try:
            video, audio = self.read_video(self.file_list[idx][0], float(self.file_list[idx][1]))
            indices = np.arange(2, video.size()[0], 6)
            frame_buffer = []
            for i in indices:
                frame = video[i].permute(2, 0, 1)
                frame = self.transforms(frame)
                frame_buffer.append(frame)
            #print(video.dtype)
            frame_idx = random.randint(0, len(frame_buffer)-1)
            frame = frame_buffer[frame_idx]
            #print(frame.dtype)
            #print("after transform", frame.dtype)
        except Exception as e:
            print("read video error", file=sys.stderr)
            print(self.file_list[idx], e, file=sys.stderr)
            other_idx = random.randint(0, self.__len__() - 2)
            if other_idx == idx:
                other_idx = self.__len__() - 1
            self.file_list[idx] = self.file_list[other_idx]
            return self.__getitem__(idx)

        _, false_audio = self.read_video(self.file_list[idx][2], float(self.file_list[idx][3]))
        try:
            save_to_path = Path(self.cache_dir, "{}.pkl".format(idx))
            with  save_to_path.open("wb+") as save_to_file:
                pickle.dump((self.file_list[idx], frame_buffer, audio, false_audio), save_to_file)
        except Exception as e:
            print("dump to cache error", file=sys.stderr)
            print(self.file_list[idx], e, file=sys.stderr)
        return frame, audio, false_audio
    
class InvalidVideo(Exception):
    pass

if __name__ == "__main__":
    local_dir = r""
    file_list_filename = ""
    remote_dir = ""
    server = ""
    username = ""
    secret_filename = ""
    top = 0
    cache_dir = Path(local_dir, "cache_loader")
    from util import audio_transform_mel
    dataset = flickr_dataset(local_dir, file_list_filename, remote_dir, server, username, secret_filename, top, cache_dir, audio_transform_mel)
    i = 120
    sample_v, sample_a, false_sample_a = dataset.__getitem__(i)
    print(sample_a.shape)
    print(sample_v.dtype, sample_a.dtype)
    '''from datetime import datetime as dt
    for i in range(1260, 1261):
        start = dt.now()
        sample_v, sample_a, false_sample_a = dataset.__getitem__(i)
        end = dt.now()
        print(end-start)
        start = dt.now()
        sample_v, sample_a, false_sample_a = dataset.__getitem__(i)
        end = dt.now()
        print(end-start)'''
    from matplotlib import pyplot as plt
    from matplotlib import cm
    # subplot(r,c) provide the no. of rows and columns
    fig, axarr = plt.subplots(1,3) 
    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    a_img = sample_a.squeeze(0).numpy()
    false_a_img = false_sample_a.squeeze(0).numpy()
    origin = "lower"
    axarr[0].imshow(a_img, cmap='viridis', origin=origin)
    axarr[1].imshow(false_a_img, cmap='viridis', origin=origin)
    axarr[2].imshow(a_img - false_a_img, cmap='viridis', origin=origin)
    fig.colorbar(cm.ScalarMappable(), ax=axarr)
    plt.show()
    #sample_img = torchvision.transforms.ToPILImage()(sample_v)
    #sample_img.show()
    
