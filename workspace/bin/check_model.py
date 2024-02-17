import glob

MODEL_PATH = "../clover-line/"
PROJECT = "Math"
VERSION_FILE = "collectMath.txt"

SEED = [i for i in range(10, 20, 1)]

version_list = []
vf = open(VERSION_FILE, "r")
while True:
    line = vf.readline()
    if line:
        version_list.append(line.split(" ")[1].replace("\n", ""))
    else:
        break

lack_seed_list = []
for v in version_list:
    for s in SEED:
        to_model_path = MODEL_PATH + PROJECT + "/" + str(v) + "/MODEL/" + str(s) + "/"
        model_name_list = glob.glob(to_model_path + "model_*")
        if len(model_name_list) == 0:
            lack_seed_list.append(v)
            break
print(lack_seed_list)
