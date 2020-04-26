from get_file_list import get_file_list
from shuffle import shuffle
from fix_window import fix_window
import sys

if __name__ == "__main__":
    folder = sys.argv[1]
    prefix = sys.argv[2]
    all_filename = "{}.txt".format(prefix)
    total = get_file_list(folder, all_filename)
    train_num = round(total * 0.95)
    train_filename = "{}_train.txt".format(prefix)
    test_filename = "{}_test.txt".format(prefix)
    shuffle(all_filename, train_filename, train_num, test_filename)
    train_fixed_filename = "{}_train_fixed.txt".format(prefix)
    test_fixed_filename = "{}_test_fixed.txt".format(prefix)
    fix_window(train_filename, train_fixed_filename)
    fix_window(test_filename, test_fixed_filename)
