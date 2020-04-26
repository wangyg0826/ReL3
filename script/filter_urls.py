import sys
import urllib.request
import os

def filter(url):
    request = urllib.request.Request(url)
    request.add_header("User-Agent", r"Mozilla/5.0 (Windows NT 6.3; WOW64; rv:52.0) Gecko/20100101 Firefox/52.0")
    try:
        response = urllib.request.urlopen(request)
        return response.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception as e:
        return -1

if __name__ == "__main__":
    found_save_point = False
    try:
        with open(sys.argv[2], 'rb') as output_file:
            output_file.seek(-2, os.SEEK_END)
            while output_file.read(1) != b'\n':
                output_file.seek(-2, os.SEEK_CUR) 
            last_line = output_file.readline().decode().strip()
    except Exception as e:
        print(e)
        found_save_point = True
    with open(sys.argv[1], 'r') as input_file:
        with  open(sys.argv[2], 'a+') as output_file:
            with open(sys.argv[3], 'a+') as error_file:
                count_yes = 0
                count_no = 0
                for line in input_file:
                    line = line.strip()
                    if not found_save_point:
                        if line == last_line:
                            found_save_point = True
                    else:
                        if line:
                            status = filter(line)
                            do_flush = False
                            if status == 200:
                                count_yes += 1
                                if count_yes % 500 == 0:
                                    do_flush = True
                                    count_yes = 0
                                print(line, file=output_file, flush=do_flush)
                            else:
                                count_no += 1
                                if count_no % 500 == 0:
                                    do_flush = True
                                    count_no = 0
                                print(line, file=error_file, flush=do_flush)