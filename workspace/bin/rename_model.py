import glob
import shutil
import re

CLOVER_LINE_PATH = "../clover-line/"
PROJECT = "totinfo/"
MODEL_PATH = "/MODEL/"
SEED = [i for i in range(20, 30, 1)]
VERSION_FILE = "collecttotinfo.txt"

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

version_list = []
vf = open(VERSION_FILE, "r")
while True:
    line = vf.readline()
    if line:
        version_list.append(line.split(" ")[1].replace("\n", ""))
    else:
        break

#version_list = version_list[0:1]   #テスト用
for v in version_list:
    print("\n" + v)
    for s in SEED:
        print(s, end="")
        model_name_list = []
        to_model_path = CLOVER_LINE_PATH + PROJECT + v + MODEL_PATH + str(s) + "/"
        model_name_list = sorted(glob.glob(to_model_path + "model_*"), key=natural_keys)
        if len(model_name_list) == 0:
            break
        first_model_path = model_name_list[0]
        last_model_path = model_name_list[-1]

        shutil.copytree(first_model_path, to_model_path + "model_first")
        shutil.copytree(last_model_path, to_model_path + "model_last")


#print(model_name_list)
