import os
import sys
import argparse
import glob

parser = argparse.ArgumentParser(description='online classification execution traces.')
parser.add_argument('project_id', help='project ID of defects4j')
parser.add_argument('bug_id', help='bug ID of defects4j')
args = parser.parse_args()

PROJECT = args.project_id
VERSION = args.bug_id.replace("\r", "")
PASS_COUNT_REPORT_PATH = "{}/coverage-v{}/pass/".format(PROJECT, VERSION)
FAIL_COUNT_REPORT_PATH = "{}/coverage-v{}/fail/".format(PROJECT, VERSION)

os.mkdir("{}/coverage-v{}/line".format(PROJECT, VERSION))
os.mkdir("{}/coverage-v{}/line/{}".format(PROJECT, VERSION, VERSION))
OUTPUT_ROOT_PATH = "{}/coverage-v{}/line/{}/".format(PROJECT, VERSION, VERSION)
COVKEYSTR = "        -:"
LINE_KEY_STR = "    #####:   34:"

def make_is_plane(output_path, input_path):
    rf = open(input_path, "r")
    #print(input_path, output_path)
    wf = open(output_path, "w")
    while True:
        line = rf.readline()
        if line:
            line_flag = line[0:len(COVKEYSTR)]
            if "-" in line_flag:
                wf.write("COMMENT\n")
            else:
                line_wr = line[len(LINE_KEY_STR):]
                wf.write(line_wr)
        else:
            break
    rf.close()
    wf.close()
    return

def make_count_coverage(output_path, input_path):
    rf = open(input_path, "r")
    #print(input_path)
    wf = open(output_path, "w")

    while True:
        line = rf.readline()
        if line:
            line_flag = line[0:len(COVKEYSTR)]
            if "-" in line_flag:
                pass
            elif "#####" in line_flag:
                pass
            else:
                execute_line = line[len(LINE_KEY_STR):]
                wf.write(execute_line)
        else:
            break
    rf.close()
    wf.close()
    return

pass_file_name_list = glob.glob(os.path.join(PASS_COUNT_REPORT_PATH, "*.log"))
fail_file_name_list = glob.glob(os.path.join(FAIL_COUNT_REPORT_PATH, "*.log"))
#print(len(pass_file_name_list), len(fail_file_name_list))
make_is_plane(OUTPUT_ROOT_PATH + "is_plane.txt", pass_file_name_list[0])  ###create is_palne.txt
for name_ind, filename in enumerate(pass_file_name_list):
    output_filename = "pass_{}.csv".format(str(name_ind + 1).zfill(4))
    make_count_coverage(OUTPUT_ROOT_PATH + output_filename, filename)

for name_ind, filename in enumerate(fail_file_name_list):
    output_filename = "fail_{}.csv".format(str(name_ind + 1).zfill(4))
    make_count_coverage(OUTPUT_ROOT_PATH + output_filename, filename)
