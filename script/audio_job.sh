#!/bin/bash
# Name and email settings - replace myuniqname with your uniqname
#SBATCH --job-name=L3
#SBATCH --mail-user=wangyg@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Resources requested - change partition to "gpu" if you want GPUs instead of CPUs
#SBATCH --partition=standard
#SBATCH --account=engin1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4g
#SBATCH --time=12:00:00


# Logging details (file locations to save console output) - replace myuniqname with your uniqname
#SBATCH --output=/home/wangyg/L3/%x_out_%j.log
#SBATCH --error=/home/wangyg/L3/%x_err_%j.log

###### Commands to run for the job go below - everything below this is just an example #####

train_task="audio" # training and evaluation subtask

bs=170 # batch size
#lr="1e-4" # learning rate
#e=3 # epochs
#seq_length=64 # max sequence length

#logging_steps=25 # how often to log
#save_steps=0 # how often to save checkpoint
eval_bs=170 # evaluation batch size

file_list="/scratch/engin_root/engin1/wangyg1/ESC-50/files.txt" # file list
out_dir="/home/wangyg/L3" # output dir name
epochs=50
data_dir="/scratch/engin_root/engin1/wangyg1/ESC-50/"
cache_dir="/scratch/engin_root/engin1/wangyg1/ESC-50/cache"
num_workers=3
l3_model="/home/wangyg/L3/l3_model_4k.model"

#module load python3.7-anaconda/2019.07
#module load ffmpeg/3.2.4
module load cuda/10.1.105
module load cudnn/10.1-v7.6
source activate L3

# train, eval, and test on train_task
python3 /home/wangyg/gitrepos/L3/model/train_audio.py ${out_dir} --local_dir=${data_dir} --cache_dir=${cache_dir} --batch_size=${bs} --validation_batch_size=${eval_bs} --file_list=${file_list} --epochs=${epochs} --num_workers=${num_workers} --pretrained_model=${l3_model} --verbose --shuffle
