#!/bin/bash
# Name and email settings - replace myuniqname with your uniqname
#SBATCH --job-name=L3
#SBATCH --mail-user=wangyg@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Resources requested - change partition to "gpu" if you want GPUs instead of CPUs
#SBATCH --partition=gpu
#SBATCH --account=engin1
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=20g
#SBATCH --time=12:00:00
#SBATCH --gpus=1

# Logging details (file locations to save console output) - replace myuniqname with your uniqname
#SBATCH --output=/home/wangyg/L3/%x_out_%j.log
#SBATCH --error=/home/wangyg/L3/%x_err_%j.log

###### Commands to run for the job go below - everything below this is just an example #####

train_task="L3" # training and evaluation subtask

bs=14 # batch size
#lr="1e-4" # learning rate
#e=3 # epochs
#seq_length=64 # max sequence length

#logging_steps=25 # how often to log
#save_steps=0 # how often to save checkpoint
#eval_bs=64 # evaluation batch size

server=""
username=""
secret=""
remote_dir=""
file_list="" # file list
out_dir="" # output dir name
tmp_dir=""
epochs=10

echo ${tmp_dir}

module load python3.7-anaconda/2019.07
module load ffmpeg/3.2.4
module load cuda/10.1.105
module load cudnn/10.1-v7.6
source activate L3

# train, eval, and test on train_task
python3 /home/wangyg/L3/train.py ${out_dir} --local_dir=${tmp_dir} --batch_size=${bs} --file_list=${file_list} --server=${server} --username=${username} --secret=${secret} --remote_dir=${remote_dir} --epochs=${epochs} --verbose --shuffle
