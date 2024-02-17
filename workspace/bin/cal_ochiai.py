import glob
import argparse
import os
import math
import copy

parser = argparse.ArgumentParser(description='online classification execution traces.')
parser.add_argument('project_id', help='project ID of defects4j')
parser.add_argument('bug_id', help='bug ID of defects4j')
args = parser.parse_args()

######################################operation val######################################

TRACE_PATH = "../clover/"
TRACE_LINE_PATH = "../clover-line/"
PROJECT = args.project_id
VERSION_PROJECT = args.bug_id
TRACE_VERSION =  TRACE_PATH + PROJECT + "/" + VERSION_PROJECT
TRACE_LINE_VERSION = TRACE_LINE_PATH + PROJECT + "/" + VERSION_PROJECT

WRITE_O = "ochiai1030.log"
WRITE_TR = "tarantula1030.log"
WRITE_SEQ_O = "seq_sus_ochiai/"
WRITE_SEQ_TR = "seq_sus_tara/"

NP = []  #the number of times a statement is not executed (e) in passed tests.
NF = []  #the number of times a statement is not executed (e) in failed tests.
EP = []   #the number of times a statement is executed (e) in passed tests.
EF = []   #the number of times a statement is executed (e) in failed tests.
PROB_OCHIAI = []   #故障疑わしさスコアを格納するリスト
PROB_TR = []
LOC = 0
passn = 0
failn = 0
SRC_LINE = []
CHECKED_LINE_LIST = []
CHECKED_LINE_LIST_STR = []

coverage_file_list =  glob.glob(os.path.join(TRACE_VERSION + "/", "*.csv"))
testn = len(coverage_file_list)
for cf in coverage_file_list:
    if "fail" in cf:
        failn += 1
    elif "pass" in cf:
	    passn += 1
plane_f = open(TRACE_VERSION + "/is_plane.txt", "r")
plane_line_f = open(TRACE_LINE_VERSION + "/is_plane.txt", "r")
while True:
    line = plane_f.readline()
    src_line = plane_line_f.readline()
    if src_line != "COMMENT\n":
        SRC_LINE.append(src_line)
    if line:
        if "0" in line:
            LOC += 1
    else:
        break
plane_f.close()
plane_line_f.close()
SRC_LINE_COPY = copy.deepcopy(SRC_LINE)

NP = [0] * LOC
NF = [0] * LOC
EP = [0] * LOC
EF = [0] * LOC
print("LOC: ", LOC)

for cf in coverage_file_list:
    read_line_cnt = 0
    coverage = open(cf, "r")
    while True:
        line = coverage.readline()
        if line:
            if "0" in line:
                if "fail" in cf:
                    NF[read_line_cnt] += 1
                else:
                    NP[read_line_cnt] += 1
            elif "1" in line:
                if "fail" in cf:
                    EF[read_line_cnt] += 1
                else:
                    EP[read_line_cnt] += 1 
            else:
                print("ERROR, unexpected values.", line)
                break
            read_line_cnt += 1
        else:
            break
    coverage.close()

###check
for check_ind, check in enumerate(NP):
    if NP[check_ind] + EP[check_ind] != passn:
        print("ERROR, NP + EP != pass total")
    if NF[check_ind] + EF[check_ind] != failn:
        print("ERROR, NF + EF != fail total")
print("NP, NF, EP, EF are created.")

def cal_ochiai(np, nf, ep, ef):
    prob_per_line = [0] * LOC
    for lind, l in enumerate(np):
        if ef[lind] + ep[lind] == 0:
            prob_per_line[lind] = 0
        else:
            bunbo = math.sqrt((ef[lind] + nf[lind])*(ef[lind] + ep[lind]))
            prob_per_line[lind] = ef[lind] / bunbo
    return prob_per_line

def cal_tr(np, nf, ep, ef):
    prob_per_line = [0] * LOC
    for lind, l in enumerate(np):
        if ef[lind] + ep[lind] == 0:
            prob_per_line[lind] = 0
        else:
            bunbo = (ef[lind] / (ef[lind] + nf[lind])) + (ep[lind] / (ep[lind] + np[lind]))
            prob_per_line[lind] = (ef[lind] / (ef[lind] + nf[lind])) / bunbo
    return prob_per_line

PROB_OCHIAI = cal_ochiai(NP, NF, EP, EF)
PROB_TR = cal_tr(NP, NF, EP, EF)

##TODO: 全く同じステートメントが存在するか判定→ある場合はそのインデックスを収集→TopN ％の計算時に分子を追加しないの処理の実装
for src_ind, src_l in enumerate(SRC_LINE):
    if src_l in CHECKED_LINE_LIST_STR:
        pass
    else:
        same_line_index = [i for i, x in enumerate(SRC_LINE_COPY) if x == src_l]
        #print(same_line_index)
        if len(same_line_index) == 1:
            pass
        else:
            """
            same_ochiai_score_list = [PROB_OCHIAI[i] for i in same_line_index]
            same_tr_score_list = [PROB_TR[i] for i in same_line_index]
            max_score_ochiai = max(same_ochiai_score_list)
            max_score_tr = max(same_tr_score_list)
            for i in same_line_index:
                PROB_OCHIAI[i] = max_score_ochiai
                PROB_TR[i] = max_score_tr
            """
            CHECKED_LINE_LIST.append(same_line_index)
            CHECKED_LINE_LIST_STR.append(src_l)
#print(PROB_OCHIAI, PROB_TR)

fail_prb_ochiai = dict()
fail_prb_tr = dict()
plot_x_bug_p = []
plot_x_bug_f = []
for i in range(LOC):
    fail_prb_ochiai[i] = 0
    fail_prb_tr[i] = 0
    plot_x_bug_p.append(0)
    plot_x_bug_f.append(0)

for cf in coverage_file_list:
    read_line_cnt = 0
    if "fail" in cf:
        cv_f = open(cf, "r")
        while True:
            line = cv_f.readline()
            if line:
                if "1" in line:
                    fail_prb_ochiai[read_line_cnt] = PROB_OCHIAI[read_line_cnt]
                    fail_prb_tr[read_line_cnt] = PROB_TR[read_line_cnt]
                else:
                    pass
                read_line_cnt += 1
            else:
                break
        cv_f.close()

from ablationex_util import calc_topn as caltopn
chunk_path = "../chunks/" + PROJECT.lower() + "_" + VERSION_PROJECT + "_buggy_chunks.yaml"
plane_path = TRACE_VERSION + "/is_plane.txt"
bug_line_id_list, num_of_bug = caltopn.get_line_id_bug(chunk_path, plane_path)

sorted_ochiai = dict(sorted(fail_prb_ochiai.items(), key=lambda x:x[1], reverse=True))
sorted_tr = dict(sorted(fail_prb_tr.items(), key=lambda x:x[1], reverse=True))

topnp_ochiai = caltopn.cal_topn_p(sorted_ochiai, bug_line_id_list, LOC, num_of_bug, CHECKED_LINE_LIST)
topnp_tr = caltopn.cal_topn_p(sorted_tr, bug_line_id_list, LOC, num_of_bug, CHECKED_LINE_LIST)
print(topnp_ochiai, topnp_tr)
print(CHECKED_LINE_LIST)
print("Line: EP EF")
plot_y = []
for ep_ind, ep in enumerate(EP):
    plot_y.append(ep_ind + 1)
    print(ep_ind + 1, ": ", ep, " ", EF[ep_ind])

"""
import numpy as np
import matplotlib.pyplot as plt
from ablationex_util import calc_topn as caltopn
chunk_path = "../chunks/" + PROJECT.lower() + "_" + VERSION_PROJECT + "_buggy_chunks.yaml"
plane_path = TRACE_VERSION + "/is_plane.txt"
bug_line_id_list, num_of_bug = caltopn.get_line_id_bug(chunk_path, plane_path)
for bind_list in bug_line_id_list:
    for bind in bind_list:
        plot_x_bug_p[bind] = EP[bind]
        plot_x_bug_f[bind] = EF[bind]
        EP[bind] = 0
        EF[bind] = 0

plot_y_np = np.array(plot_y)
plot_x_bug_p_np = np.array(plot_x_bug_p)
plot_x_bug_f_np = np.array(plot_x_bug_f)
plot_x_np = np.array(EP)
plot_x_fp = np.array(EF)

plt.xlabel("Line Number of Code")
plt.ylabel("Execution Count")
#plt.ylabel("Suspiciousness Score")
plt.title("Execution Count in Pass Test Case")
#plt.title("")
#plt.bar(plot_y_np, plot_x_np, color="blue")
#plt.savefig("pass_count.png")
#plt.ylim(0, 15)
plt.title("Execution Count in Pass Test Case")
plt.bar(plot_y_np, plot_x_np, color="blue")
plt.bar(plot_y_np, plot_x_bug_p_np, color="red")
plt.savefig("images-susscore/multi/pass_count.png")
"""
#'''
ochiai_o = open(WRITE_O, "a")
tr_o = open(WRITE_TR, "a")
ochiai_o.write(PROJECT + " " + VERSION_PROJECT + " " + str(topnp_ochiai) + "\n")
tr_o.write(PROJECT + " " + VERSION_PROJECT + " " + str(topnp_tr) + "\n")
ochiai_o.close()
tr_o.close()
#'''
'''
def write_seq_sus(seqf, tr_dict):
    #seqf.write(PROJECT + " " + VERSION_PROJECT + "\n")
    for ln, susp in tr_dict.items():
        seqf.write(str(ln) + ":" + str(susp) + "\n")
    #seqf.write("COMPLETE\n\n")
    return
seq_o = open(WRITE_SEQ_O + PROJECT + "_" + VERSION_PROJECT + ".log", "w")
seq_tr = open(WRITE_SEQ_TR + PROJECT + "_" + VERSION_PROJECT + ".log", "w")
write_seq_sus(seq_o, fail_prb_ochiai)
write_seq_sus(seq_tr, fail_prb_tr)
seq_o.close()
seq_tr.close()
'''