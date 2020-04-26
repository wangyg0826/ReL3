import sys
import random

def shuffle(input_filename, output_filename_1, num_1=0, output_filename_2=None):
    with open(input_filename, "r") as input_file:
        lines = input_file.readlines()
    with open(output_filename_1, "w+") as output_file:
        if num_1 > 1:
            k = num_1
        else:
            k = len(lines)
        sample = random.sample(lines, k)
        output_file.writelines(sample)
    remain = len(lines) - k
    if remain > 0 and output_filename_2 is not None:
        with open(output_filename_2, "w+") as output_file:
            sample = random.sample(lines, remain)
            output_file.writelines(sample)

if __name__ == "__main__":
    if len(sys.argv) > 4:
        shuffle(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
    elif len(sys.argv) > 3:
        shuffle(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    else:
        shuffle(sys.argv[1], sys.argv[2])
