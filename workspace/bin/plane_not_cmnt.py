
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='online classification execution traces.')
parser.add_argument('project_id', help='project ID of defects4j')
parser.add_argument('bug_id', help='bug ID of defects4j')
args = parser.parse_args()

######################################operation val######################################
TRACE_PATH = "../clover-line/"
PROJECT = args.project_id
VERSION_PROJECT = args.bug_id
TRACE_VERSION =  TRACE_PATH + PROJECT + "/" + VERSION_PROJECT

plane_f = open(TRACE_VERSION + "/is_plane.txt", "r")
w_notcm = open(TRACE_VERSION + "/not_cmnt_plane.txt", "w")

while True:
    line = plane_f.readline()
    if line:
        if line == "COMMENT\n":
            pass
        else:
            w_notcm.write(line)
    else:
        break

plane_f.close()
w_notcm.close()