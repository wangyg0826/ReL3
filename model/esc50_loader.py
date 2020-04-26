import  numpy as np
from pathlib import Path
import av
import os
from torch.utils.data import Dataset
import pickle
import torch
import sys
import librosa


class esc50_dataset(Dataset):

    def read_full_audio_and_transform(self, filename):
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
        audio = np.mean(audio, axis=0, keepdims=False)
        buffer = []
        num_subclips = 10
        for i in range(num_subclips):
            start_idx = round(float(i) / (num_subclips - 1) * (total_audio_frames-afps))
            end_idx = min(start_idx + afps, total_audio_frames)
            audio_one_sec = audio[start_idx:end_idx]
            audio_one_sec = self.audio_transform_func(audio_one_sec)
            buffer.append(audio_one_sec)
        return buffer

    def __init__(self, parent_dir, file_list_filename, top_k, cache_dir, include_partitions, audio_transform_func):
        count = 0
        self.label_names = dict()
        next_label_num = 0
        self.cache_dir = cache_dir
        self.parent_dir = parent_dir
        self.file_list = []
        self.audio_transform_func = audio_transform_func
        labels = []
        with open(file_list_filename, 'r') as file_list_file:
            for data_filename in file_list_file:
                data_filename = data_filename.strip()
                if data_filename:
                    try:
                        folder, name = data_filename.split(r"/")
                        #data_file_path = os.path.join(parent_dir, folder, name)
                        partition = np.int32(name[0])
                        if partition in include_partitions:
                            if top_k > 0:
                                if count < top_k:
                                    count += 1
                                else:
                                    break
                            if folder not in self.label_names:
                                self.label_names[folder] = next_label_num
                                next_label_num += 1
                            label = self.label_names[folder]
                            label_array = np.empty(10, dtype=np.int32)
                            label_array.fill(label)
                            labels.extend(label_array)
                            self.file_list.append((folder, name))
                    except Exception as e:
                        print(data_filename, e, file=sys.stderr)
        self.labels = np.asarray(labels)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        file_list_idx, slice_idx = np.divmod(idx, 10)
        stem = Path(self.file_list[file_list_idx][1]).stem
        read_from_path = Path(self.cache_dir, "{}_{}.pkl".format(self.file_list[file_list_idx][0], stem))
        if read_from_path is not None and read_from_path.exists():
            try:
                with read_from_path.open("rb") as read_from:
                    buffer = pickle.load(read_from)
                    #print(len(buffer))
                    return buffer[slice_idx], self.labels[idx]
                read_from_path.unlink()   
            except Exception as e:
                print("load from cache error", file=sys.stderr)
                print(self.file_list[file_list_idx], e, file=sys.stderr)
        filename = os.path.join(self.parent_dir, self.file_list[file_list_idx][0], self.file_list[file_list_idx][1])
        buffer = self.read_full_audio_and_transform(filename)
        try:
            with read_from_path.open("wb+") as save_to_file:
                pickle.dump(buffer, save_to_file)
        except Exception as e:
            print("dump to cache error", file=sys.stderr)
            print(self.file_list[file_list_idx], e, file=sys.stderr)
        return buffer[slice_idx], self.labels[idx]





if __name__ == "__main__":
    parent_dir = r".\data\ESC-50"
    cache_dir = r".\data\ESC-50\cache_loader"
    file_list_filename = r".\data\ESC-50\files.txt"
    include_partitions = [1, 2, 3, 4, 5]
    include_partitions = set(include_partitions)
    top_k = 200
    from util import audio_transform_mel
    esc50ds = esc50_dataset(parent_dir, file_list_filename, top_k, cache_dir, include_partitions, audio_transform_mel)
    audio, label = esc50ds.__getitem__(998)
    #print(esc50ds.labels.shape, esc50ds.labels.dtype)
    print(audio.shape, audio.dtype)
    from matplotlib import pyplot as plt
    audio = audio.squeeze(0).numpy()
    origin = "lower"
    plt.figure(figsize=(10, 4))
    plt.imshow(audio, origin=origin)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.show()
