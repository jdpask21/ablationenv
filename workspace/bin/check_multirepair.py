import argparse
from ablationex_util import calc_topn as caltopn

parser = argparse.ArgumentParser(description='online classification execution traces.')
parser.add_argument('project_id', help='project ID of defects4j')
parser.add_argument('bug_id', help='bug ID of defects4j')
args = parser.parse_args()

######################################operation val######################################

TRACE_PATH = "../clover-line/"
PROJECT = args.project_id
VERSION_PROJECT = args.bug_id.replace("\r", "")
TRACE_VERSION =  TRACE_PATH + PROJECT + "/" + VERSION_PROJECT

#print(PROJECT, VERSION_PROJECT)
check_flag = caltopn.check_multirepair("../chunks/" + "{}_{}_buggy_chunks.yaml".format(PROJECT.lower(), VERSION_PROJECT))
if check_flag:
    print(PROJECT, VERSION_PROJECT)