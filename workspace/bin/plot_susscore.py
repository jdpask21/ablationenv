TARGET_SUS_PATH = "seq_sus_ochiai/"

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='online classification execution traces.')
parser.add_argument('project_id', help='project ID of defects4j')
parser.add_argument('bug_id', help='bug ID of defects4j')
args = parser.parse_args()

######################################operation val######################################
TRACE_PATH = "../clover/"
PROJECT = args.project_id
VERSION_PROJECT = args.bug_id
TRACE_VERSION =  TRACE_PATH + PROJECT + "/" + VERSION_PROJECT
SAVE_FIG_PATH = "images-susscore/{}{}-o2.png".format(PROJECT, VERSION_PROJECT)

read_file_path = "{}{}_{}.log".format(TARGET_SUS_PATH, PROJECT, VERSION_PROJECT)
sus_scoref = open(read_file_path, "r")
plot_y = []
plot_x = []
plot_y_bug = []
index_line = 0
not_zero_socre_l = []
not_zero_socre_d = dict()

while True:
    y_line = sus_scoref.readline()
    if y_line:
        line_int = int(y_line.split(":")[0])
        score_float = float(y_line.split(":")[1].replace("\n", ""))
        if score_float != 0:
            not_zero_socre_d[index_line] = score_float
            not_zero_socre_l.append(score_float)
        index_line += 1
        plot_x.append(line_int)
        plot_y.append(score_float)
        plot_y_bug.append(0)
    else:
        break
sus_scoref.close()

plot_y_tensor = torch.tensor(not_zero_socre_l)
m = torch.nn.Softmax(dim=0)
not_zero_tensor_sf = m(plot_y_tensor)
not_zero_sf_np = not_zero_tensor_sf.numpy()
fill_ind = 0
for i_ind, i in enumerate(plot_y):
    if i_ind in not_zero_socre_d:
        plot_y[i_ind] = not_zero_sf_np[fill_ind]
        fill_ind += 1

from ablationex_util import calc_topn as caltopn
chunk_path = "../chunks/" + PROJECT.lower() + "_" + VERSION_PROJECT + "_buggy_chunks.yaml"
plane_path = TRACE_VERSION + "/is_plane.txt"
bug_line_id_list, num_of_bug = caltopn.get_line_id_bug(chunk_path, plane_path)
for bind_list in bug_line_id_list:
    for bind in bind_list:
        plot_y_bug[bind] = plot_y[bind]
        plot_y[bind] = 0

plot_y_np = np.array(plot_y)
plot_y_bug_np = np.array(plot_y_bug)
plot_x_np = np.array(plot_x)
# x 軸のラベルを設定する。
plt.xlabel("Line Number of Code")
# y 軸のラベルを設定する。
plt.ylabel("Suspiciousness Score")
# タイトルを設定する。
plt.title("Suspiciousness Score of Ochiai")
#print(plot_y_sf_np)
plt.bar(plot_x_np, plot_y_np, color="blue")
plt.bar(plot_x_np, plot_y_bug_np, color="red")
plt.savefig(SAVE_FIG_PATH)
plt.show()

