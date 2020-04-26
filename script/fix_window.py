import random
import sys

def fix_window(input_filename, output_filename):
    filenames = []
    with open(input_filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                filenames.append(line)
    length = len(filenames)
    pairs = []
    for i in range(length):
        filename = filenames[i]
        timepoint = random.uniform(0.0, 1.0)
        negative_idx = random.randint(0, length - 2)
        if negative_idx == i:
            negative_idx = length - 1
        timepoint_neg = random.uniform(0.0, 1.0)
        line = "{};{};{};{}\n".format(filename, timepoint, filenames[negative_idx], timepoint_neg)
        pairs.append(line)
    with open(output_filename, 'w+') as f:
        f.writelines(pairs)

if __name__ == "__main__":
    fix_window(sys.argv[1], sys.argv[2])
