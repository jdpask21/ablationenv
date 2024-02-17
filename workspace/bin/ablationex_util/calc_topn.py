def get_line_of_code(path):
    plane = open(path, "r")
    loc = 0
    while True:
        line = plane.readline()
        if line:
            if line != "COMMENT\n":
                loc += 1
            else:
                pass
        else:
            break
    plane.close()
    return loc

def get_unique_list(seq):
    seen = []
    return [x for x in seq if x not in seen and not seen.append(x)]
'''
def get_line_of_bug(chunk_path, plane_path):
    tmp_line_numbers = []
    tmp_chunk = 1
    lines_of_bug = []
    line_numbers = []
    chunk = open(chunk_path, "r")
    num_of_bug = 0
    while True:   ###バグの行番号の獲得
        line = chunk.readline()
        if line:
            num_of_bug = int(line.split(":")[1].replace(" ", "").replace("\n", "")) + 1
            if num_of_bug != tmp_chunk and num_of_bug != 1:
                line_numbers.append(tmp_line_numbers)
                tmp_line_numbers = []
                tmp_chunk = num_of_bug
            tmp_line_numbers.append(int(line.split(":")[0]))
        else:
            break
    line_numbers.append(tmp_line_numbers)
    chunk.close()
    plane = open(plane_path, "r")
    line_cnt = 0
    lines_of_bug = [[] for i in range(len(line_numbers))]
    while True:   ###バグの行（文字情報）を獲得
        line = plane.readline()
        if line:
            line_cnt += 1
            for lmd, line_numbers_l in enumerate(line_numbers):
                if line_cnt in line_numbers_l:
                   lines_of_bug[lmd].append(line)
        else:
            break
    plane.close()
    lines_of_bug = get_unique_list(lines_of_bug)
    return lines_of_bug, len(lines_of_bug)   ###全く同じ内容の行情報は削除(全く同じコード（行）が複数存在し，多重故障している場合は提案手法は非対応)
'''
def get_line_of_bug(chunk_path, plane_path):
    tmp_line_numbers = []
    tmp_chunk = 1
    lines_of_bug = []
    line_numbers = []
    chunk = open(chunk_path, "r")
    num_of_bug = 0
    while True:   ###バグの行番号の獲得
        line = chunk.readline()
        if line:
            num_of_bug = int(line.split(":")[1].replace(" ", "").replace("\n", "")) + 1
            if num_of_bug != tmp_chunk and num_of_bug != 1:
                line_numbers.append(tmp_line_numbers)
                tmp_line_numbers = []
                tmp_chunk = num_of_bug
            tmp_line_numbers.append(int(line.split(":")[0]))
        else:
            break
    line_numbers.append(tmp_line_numbers)
    chunk.close()
    plane = open(plane_path, "r")
    line_cnt = 0
    lines_of_bug = [[] for i in range(len(line_numbers))]
    while True:   ###バグの行（文字情報）を獲得
        line = plane.readline()
        if line:
            line_cnt += 1
            for lmd, line_numbers_l in enumerate(line_numbers):
                if line_cnt in line_numbers_l:
                   lines_of_bug[lmd].append(line)
        else:
            break
    plane.close()
    lines_of_bug = get_unique_list(lines_of_bug)
    return lines_of_bug, len(lines_of_bug)   ###全く同じ内容の行情報は削除(全く同じコード（行）が複数存在し，多重故障している場合は提案手法は非対応)

def get_line_id_bug(chunk_path, plane_path):
    tmp_line_numbers = []
    tmp_chunk = 1
    line_numbers = []
    id_list = []
    chunk = open(chunk_path, "r")
    num_of_bug = 0
    while True:   ###バグの行番号の獲得
        line = chunk.readline()
        if line:
            num_of_bug = int(line.split(":")[1].replace(" ", "").replace("\n", "")) + 1
            if num_of_bug != tmp_chunk and num_of_bug != 1:
                line_numbers.append(tmp_line_numbers)
                tmp_line_numbers = []
                tmp_chunk = num_of_bug
            tmp_line_numbers.append(int(line.split(":")[0]))
        else:
            break
    line_numbers.append(tmp_line_numbers)
    chunk.close()
    plane = open(plane_path, "r")
    id_cnt = -1   ###0が一行目を示すため0が初期値
    line_cnt = 0
    id_list = [[] for i in range(len(line_numbers))]
    while True:   ###バグの行（文字情報）を獲得
        line = plane.readline()
        if line:
            line_cnt += 1
            if "0" in line:
                id_cnt += 1
            for lmd, line_numbers_l in enumerate(line_numbers):
                if line_cnt in line_numbers_l:
                    id_list[lmd].append(id_cnt)
        else:
            break
    plane.close()
    #print("Bug of Line: ", id_list)
    return id_list, num_of_bug

def cal_topn_p(sorted_dict, bug_list, loc, num_of_bug, same_line_list):
    ###故障箇所が全く同じ文字列の場合は、手作業でデータとる必要あり
    hit_flag = False
    hitted_list = []
    same_hitted_line_list = []
    topn_percentile = 100
    bug_number = num_of_bug
    #bug_number = num_of_bug - 2   #Lang 50
    #bug_list = bug_list[2:]   #Lang 50
    #bug_number = num_of_bug - 1   #Chart 25
    #bug_list = bug_list[:1]   #Chart 25
    inspected_line_cnt = 0
    print(bug_number, bug_list)
    for id, prob in sorted_dict.items():
        #"""
        hitted_flag = False
        for ch_s in same_line_list:
            if id in ch_s:
                hitted_flag = True
                if id in same_hitted_line_list:
                    break   ###同じステートメントで分子を追加しない処理(条件を公平にするため)
                else:
                    inspected_line_cnt += 1
                    same_hitted_line_list += ch_s
                    break
            else:
                pass
        if hitted_flag == False:
            inspected_line_cnt += 1
        #"""
        #inspected_line_cnt += 1
        for lind, linebug in enumerate(bug_list):
            if id in linebug:
                if lind not in hitted_list:
                    print(id)
                    bug_number -= 1
                    hitted_list.append(lind)
            if bug_number == 0:
                hit_flag = True
                break
        if hit_flag:
            topn_percentile = 100 * inspected_line_cnt / loc
            break
    #print(same_hitted_line_list)
    return topn_percentile

def check_unique_of_bug(chunk_path, plane_path):
    tmp_line_numbers = []
    tmp_chunk = 1
    lines_of_bug = []
    line_numbers = []
    chunk = open(chunk_path, "r")
    num_of_bug = 0
    while True:   ###バグの行番号の獲得
        line = chunk.readline()
        if line:
            num_of_bug = int(line.split(":")[1].replace(" ", "").replace("\n", "")) + 1
            if num_of_bug != tmp_chunk and num_of_bug != 1:
                line_numbers.append(tmp_line_numbers)
                tmp_line_numbers = []
                tmp_chunk = num_of_bug
            tmp_line_numbers.append(int(line.split(":")[0]))
        else:
            break
    line_numbers.append(tmp_line_numbers)
    chunk.close()
    plane = open(plane_path, "r")
    line_cnt = 0
    lines_of_bug = [[] for i in range(len(line_numbers))]
    while True:   ###バグの行（文字情報）を獲得
        line = plane.readline()
        if line:
            line_cnt += 1
            for lmd, line_numbers_l in enumerate(line_numbers):
                if line_cnt in line_numbers_l:
                   lines_of_bug[lmd].append(line)
        else:
            break
    plane.close()
    lines_of_bug_unique = get_unique_list(lines_of_bug)
    if len(lines_of_bug_unique) != len(lines_of_bug):
        print(chunk_path, " is not unique bug.")
        return True
    else:
        return False
    
def check_multirepair(chunk_path):
    chf = open(chunk_path, "r")
    tmp_chunkid = "0"
    while True:
        line = chf.readline()
        if line:
            line_number = line.split(":")[0]
            chunk_id = line.split(":")[1].replace(" ", "").replace("\n", "")
            if tmp_chunkid != chunk_id:
                chf.close()
                return True
        else:
            break
    chf.close()
    return False