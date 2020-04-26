from pathlib import Path
import sys

def get_file_list(folder, output_filename):
    p = Path(folder)
    count = 0
    with open(output_filename, 'w+') as result_file:
        for f in p.rglob("*.mp4"):
            count += 1
            result_file.write("{}\n".format(f.name))
    return count

if __name__ == "__main__":
    get_file_list(sys.argv[1], sys.argv[2])
