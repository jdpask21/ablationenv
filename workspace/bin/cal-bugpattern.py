import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ml import read_result as rl
RESULT_PATH = "./line_sfl_data/"

RESULT_LIST = ["topn_average_result_0607.log", "ochiai0607.log", "tarantula0607.log"]
SUM_TOTAL = 71
#SUM_TOTAL = 15 #Chart
#SUM_TOTAL = 28 #Math
#SUM_TOTAL = 28 #Lang
#SUM_TOTAL = 43


filename_deep = RESULT_LIST
TARGET_PROJ = ["Lang", "Chart", "Math"]
#TARGET_PROJ = [TARGET_PROJ[0], TARGET_PROJ[1]]
APPROACH = ["Ochiai", "Tarantula"]
ab_result = []
ochiai_result = []
tarantula_result = []
ochiai_margin = []
tarantula_margin = []
_ = []
project_list = []
bugid_list = []
for proj in TARGET_PROJ:
    result_v, pl, bugidl = rl.read_result_fromfile_ab(RESULT_PATH + RESULT_LIST[0], proj, proj_flag=True)
    ab_result += result_v
    project_list += pl
    bugid_list += bugidl
    result_v = rl.read_result_fromfile_ab(RESULT_PATH + RESULT_LIST[1], proj)
    ochiai_result += result_v
    result_v = rl.read_result_fromfile_ab(RESULT_PATH + RESULT_LIST[2], proj)
    tarantula_result += result_v

for fv_ind, fv in enumerate(ab_result):
    ochiai_margin.append(ochiai_result[fv_ind] - fv)
    tarantula_margin.append(tarantula_result[fv_ind] - fv)

READ_PATH = "./line_sfl_data/bug_pattern_list.log"
def get_target_bug_pattern(read_path, target_proj, target_bugid):
    bpl = open(read_path, "r")
    while True:
        line = bpl.readline()
        if line:
            proj_id = line.split(":")[0]
            if proj_id.split(" ")[0] == target_proj and proj_id.split(" ")[1] == target_bugid:
                ret_error = line.split(":")[1]
                ret_pattern = line.split(":")[2]
                ret_action = line.split(":")[3].replace("\n", "")
                break
        else:
            break
    bpl.close()
    return ret_error, ret_pattern, ret_action

def exist_is_plus_else_one(addres_dict, str_target):
    str_list = str_target.split(",")
    str_list = list(set(str_list))
    for tr_str in str_list:
        if tr_str in addres_dict:
            addres_dict[tr_str] += 1
        else:
            addres_dict[tr_str] = 1
    return

positive_error = dict()
positive_repair_pattern = dict()
positive_repair_action = dict()
negative_error = dict()
negative_repair_pattern = dict()
negative_repair_action = dict()
positive_num = 0
negative_num = 0
positive_action_num = 0
negative_action_num = 0
positive_error_num = 0
negative_error_num = 0


TARGET_EXIST_APPROACH = ochiai_margin
borderline = 0
for pind, project in enumerate(project_list):
    bugid = bugid_list[pind]
    ret_error, ret_pattern, ret_action = get_target_bug_pattern(READ_PATH, project, bugid)
    if ochiai_margin[pind] > borderline:
        exist_is_plus_else_one(positive_error, ret_error)
        exist_is_plus_else_one(positive_repair_pattern, ret_pattern)
        exist_is_plus_else_one(positive_repair_action, ret_action)
        positive_num += 1
        positive_error_num += len(ret_error.split(","))
        positive_action_num += len(ret_action.split(","))
    elif ochiai_margin[pind] < -borderline:
        exist_is_plus_else_one(negative_error, ret_error)
        exist_is_plus_else_one(negative_repair_pattern, ret_pattern)
        exist_is_plus_else_one(negative_repair_action, ret_action)
        negative_num += 1
        negative_error_num += len(ret_error.split(","))
        negative_action_num += len(ret_action.split(","))
positive_error = dict(sorted(positive_error.items(), key=lambda x:x[1], reverse=True))
positive_repair_pattern = dict(sorted(positive_repair_pattern.items(), key=lambda x:x[1], reverse=True))
positive_repair_action = dict(sorted(positive_repair_action.items(), key=lambda x:x[1], reverse=True))
negative_error = dict(sorted(negative_error.items(), key=lambda x:x[1], reverse=True))
negative_repair_pattern = dict(sorted(negative_repair_pattern.items(), key=lambda x:x[1], reverse=True))
negative_repair_action = dict(sorted(negative_repair_action.items(), key=lambda x:x[1], reverse=True))
print("Error\nPositive\n", positive_error, "\nNegative\n", negative_error)
print("Pattern\nPositive\n", positive_repair_pattern, "\nNegative\n", negative_repair_pattern)
print("Action\nPositive\n", positive_repair_action, "\nNegative\n", negative_repair_action)
print("Average Error\nPositive: ", positive_error_num / positive_num, "\nNegative: ", negative_error_num / negative_num)
print("Average Action\nPositive: ", positive_action_num / positive_num, "\nNegative: ", negative_action_num / negative_num)
print("Positive: ", positive_num, "\nNegative: ", negative_num)
