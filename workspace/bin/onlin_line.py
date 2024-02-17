from gensim.models import word2vec
from gensim.models.word2vec import Word2Vec
import matplotlib.pyplot as plt
#from sklearn.decomposition import PCA
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import torch.nn as nn
import torch.optim as optim
import torch
from random import shuffle
import random
import time
import numpy as np
import os
import copy
import sys
from ablationex_util import time_information_write as timewrite
from ablationex_util import calc_topn as caltopn
import glob
import argparse

parser = argparse.ArgumentParser(description='online classification execution traces.')
parser.add_argument('project_id', help='project ID of defects4j')
parser.add_argument('bug_id', help='bug ID of defects4j')
args = parser.parse_args()

######################################operation val######################################

TRACE_PATH = "../clover-line/"
PROJECT = args.project_id
VERSION_PROJECT = args.bug_id.replace("\r", "")
TRACE_VERSION =  TRACE_PATH + PROJECT + "/" + VERSION_PROJECT
#TRACE_VERSION = "printtoken/v5"
#TRACE_VERSION = "sed/v5-v2"
ATTENTION_ENABLE = True
OUTPUT_ATTENSION_WEIGHT = False   ###ATTENTION_ENABLEが有効時にTrueでAttention Scoreをファイルに書き出す。この変数がTrue,FAIL_TRACE_CLASSIFY=True,PASS_TR_NUM=0でプログラムが異常終了するバグあり。原因不明で修正できていない。
USING_RNN = False
BiDIRECT_LSTM = False
USING_CPU_MACHINE = True   ###CPUしか載っていないマシンで動かす場合はTrue

if BiDIRECT_LSTM:
	LSTM_OUTPUT_SIZE = 512
else:
	LSTM_OUTPUT_SIZE = 256

MLP_INPUT_IS_AVERAGE = True   ###MLPへの入力値をLSTM各層出力値の平均にする場合はTrue、最終層のみのベクトルを入力する場合はFalse

USE_EXPERIMENT_MODE = True   ###Trueにすると↓で定義されている変数の値が変わるので注意

W_V_NAME = 28   #単語ベクトルのサイズ
W_V_RETURN = 50
W_V_ARGUMENT = 50
W_V = W_V_NAME + W_V_RETURN + W_V_ARGUMENT   ###Word２Vecが出力するトータルのベクトルサイズ。三変数分割するためその合計になる。
LSTM_HIDEN_V = W_V * 2   ###LSTM隠れ層のサイズ

#'''   ###print_tokens2
FAIL_TRACES = TRACE_VERSION
REDUCED_FAIL_TRACES = TRACE_VERSION + "/reduced_traces/fail/"
PASS_TRACES = TRACE_VERSION
REDUCED_PASS_TRACES = TRACE_VERSION + "/reduced_traces/pass/"
#'''
'''   ###normal
FAIL_TRACES = TRACE_VERSION + "/ex_traces/fail/"
REDUCED_FAIL_TRACES = TRACE_VERSION + "/reduced_traces/fail/"
PASS_TRACES = TRACE_VERSION + "/ex_traces/pass/"
REDUCED_PASS_TRACES = TRACE_VERSION + "/reduced_traces/pass/"
'''
'''
FAIL_TRACES = TRACE_VERSION + "/traces/fail/"
REDUCED_FAIL_TRACES = TRACE_VERSION + "/reduced_traces/fail/"
PASS_TRACES = TRACE_VERSION + "/traces/pass/"
REDUCED_PASS_TRACES = TRACE_VERSION + "/reduced_traces/pass/"
'''
EPOCH = "/ep270_ex"
NNs_MODEL_NAME = "/model_last/"
SEED_VALUE = 20
MODEL_EPOCH = "/0425_nta_30w1_ab" + NNs_MODEL_NAME   ###model epoch

W2VEC_LEARNING = False   #Trueで学習、Falseで既存のモデルをロードする
WINDOW_ = 1
W2V_MODEL_NAME_NAME = "/wv" + str(W_V_NAME) + "-w" + str(WINDOW_) + ".model"   ###各変数におけるWord2Vecモデルのファイル名（書き込み、読み込み共用）
W2V_MODEL_NAME_RETURN = "/wv" + str(W_V_RETURN) + "-w" + str(WINDOW_) + "-return.model"
W2V_MODEL_NAME_ARGUMENTS = "/wv" + str(W_V_ARGUMENT) + "-w" + str(WINDOW_) + "-argument.model"
MODEL_SAVE_PATH = TRACE_VERSION + EPOCH + W2V_MODEL_NAME_NAME
MODEL_LOAD_PATH = TRACE_VERSION + EPOCH + W2V_MODEL_NAME_NAME

PYMODEL_PATH = TRACE_VERSION + EPOCH + MODEL_EPOCH   ###pymodelをloadする場所

WRITE_OUTPUT_FILE = True
WRITE_TIME_INFORMATION = False   ###実行時間の書き込み
CLASSIFI_LOG_NAME = "/0214-at-ab-" + NNs_MODEL_NAME[7:-1] + "-test.log"
OUTPUT_CLASSIFI = TRACE_VERSION + "/gomibako" + CLASSIFI_LOG_NAME

EACH_ATTENTION_WEIGHT_PASS_PATH = TRACE_VERSION + "/MODEL/" + str(SEED_VALUE) + "/attention_pass/"   ###それぞれのトレースにおけるAttention重みを書き込むフォルダ
EACH_ATTENTION_WEIGHT_FAIL_PATH = TRACE_VERSION + "/MODEL/" + str(SEED_VALUE) + "/attention_fail/"

####################################↓の変数は基本編集する必要がない
ABLATION_FC_LINE = True   ###True
TARGET_FC_NAME_LIST = []   ###and Ablation
SECOND_TARGET_FC_NAME_LIST = []   ###or Ablation  
SECOND_TARGET_FLAG = False   ###↑のor Ablationを有効にする場合はTrue
####################################↑上の変数は基本編集しない

ONE_LINE_TRACE_DELETE = False   ###一行のトレースを削除する
DEL_TRACE_LINE_NUM = 5   ###削除するトレースの行数指定。この値以下のトレースは削除される。5

TARGET_SH_NAME = []   ###個別シャフルの場合に使う  ###リストで指定した関数がアブレーションされる
###↓はZIKKEN modeのときは絶対Trueにしておく
SHUFFLE_TR = True   ###ここをTrueで↑の対象関数のアブレーションが実行、Falseだと実行されない
###↓はZIKKEN modeのときに切り替えできないので、固定となる。基本的にはpop=1で使うが（ターゲット飲みアブレーション）、それ以外もするには手動切り替えは必要。
POP_KAISU = 1   ###対象関数の何個後を削除するか(1はターゲット関数のみ削除)
FRONT_SAKUJO = 0   ###対象関数の何個前までを削除するか

FAIL_TRACE_CALSSIFY = True  ###TrueでFailトレースのみを分類する
PASS_TRACE_CLASSIFY = False   ###TrueでPassトレースのみを分類する
TRAINIG_DATA_ONLY_CALSSIFY = False   ###Trueで訓練データのみを分類する
TRAINING_SIZE_ = 0.3   ###↑がTrueの場合のみの、訓練データサイズの指定

STATIC_CALL_ENABLE = False  ###関数呼び出し関係を考慮する場合True
ABLATION_CALLER = []
ABLATION_CALLED = []
if STATIC_CALL_ENABLE:   ###関数呼び出し関係を考慮する場合、リストの要素がリストになるので注意
	ABLATION_TARGET_LIST_ZIKKEN = [[ABLATION_CALLER, ABLATION_CALLED]]
else:
	ABLATION_TARGET_LIST_ZIKKEN = ["original"]

END_FUNC_ABLATION = False
END_FUNC_NUM = 3

####
####実験結果に大きく影響を与えるため操作に注意 (ステートメント粒度では編集不要)
RETURN_FULL_ABL = False   ###Trueで返り値の列をアブレーションする
ARUGUMENT_FULL_ABL = False
CALL_FULL_ABL = False
####
####

##これより下のグローバル変数はステートメント粒度では編集する必要がない

PER_LINE_ABLATION = True ###Trueで行単位のアブレーションを実行
CONDITION_OUTPUT = [False, "InfoTbl"]   ###下記アブレーション条件書き込み用の変数
CONDITION_OUTPUT_PATH = TRACE_VERSION + "/MODEL/" + "{}".format(SEED_VALUE) + NNs_MODEL_NAME + "abl_cnd.log"

PER_LINE_TEST_CASE_PASS = []   ###実行するテストケース番号を格納
PER_LINE_TEST_CASE_FAIL = []

PASS_TRACES_A_to_B_ABLATION = False   ###指定した二つの数（A、B）の間に挟まる全ての行をアブレーションする場合はTrue
FAIL_TRACES_A_to_B_ABLATION = True
ABLATION_LINE_NO_PASS = []   ###何行目をアブレーションするのかをテストケース毎に指定するリストを格納するリスト
ABLATION_LINE_NO_FAIL = [[]]

ABLATION_RETURN_PASS = []   ###行単位でのアブレーション用
ABLATION_ARGUMENT_PASS = []
ABLATION_CALL_PASS = []
ABLATION_RETURN_FAIL = []
ABLATION_ARGUMENT_FAIL = []
ABLATION_CALL_FAIL = []
BLANK_STR = "0 "   ### "" or "0 "かな

GLOBAL_VAL_FILE_PATH = TRACE_VERSION + "/MODEL/" + "{}".format(SEED_VALUE) + "/abl_cnd_4ex2.log"

G_TARGET_FUNC_LIST = ["original"]
G_TARGET_FUNC = G_TARGET_FUNC_LIST[0]   ###そのままでOK（コメントアウトしないでOK）
GLOBAL_READ_FROM_FILE = False

COL_AND_ROW_ABLATION_ZIKKEN_MODE = False   ###Trueの場合G_TARGET_FUNC_LISTの中身を全て実験する。グローバル変数のファイル読み込みは必須。
SIMPLE_ABLATION_EXPERIMENT_MODE = False   ###↑がTrueの場合は起動しない。こいつがTrueの場合は↑でG_TARGET_FUNCで指定した関数に対してのみ実験を行う。上がFalseでこれもFalseの場合は何の処理も行われない。
LINE_GRANU_ABLATION_MODE = True   ###行の粒度でのアブレーションを実行するフラッグ。↑の二つのどちらかがTrueの場合はそちらが優先して実行される。
WRITE_TOPN_PERCENTILE = "topn_average_result_0214sir.log"
OUTPUT_PER_SEED = "topn_abaltion_result-0214sir.log"

##これより上のグローバル変数はステートメント粒度では編集する必要がない （ここまで）

def get_trace_num():
	failn, passn = 0, 0
	for coverage_file_path in glob.glob(os.path.join(TRACE_VERSION + "/", "*.csv")):
		if "fail" in coverage_file_path:
			failn += 1
		elif "pass" in coverage_file_path:
			passn += 1
	return failn, passn

FAIL_TR_NUM, PASS_TR_NUM = get_trace_num()
PASS_TR_NUM = 0
FAIL_TR_NUM = 10
print(FAIL_TR_NUM, PASS_TR_NUM)

if TRAINIG_DATA_ONLY_CALSSIFY and (FAIL_TRACE_CALSSIFY or PASS_TRACE_CLASSIFY):   ###グローバル変数の設定ミス
	print("TRAINIG_DATA_ONLY_CALSSIFY and (FAIL_TRACE_CALSSIFY or PASS_TRACE_CLASSIFY) is True")
	sys.exit()


#######################################start###################################
if USE_EXPERIMENT_MODE:
	TARGET_SH_NAME = []

def try_gpu(e):
    if torch.cuda.is_available():
        return e.cuda()
        #return e
    return e

def pool_trace(trace_path, target_path, num_of_tests, fail_path, target_fail, num_of_fail, returnencode, argencode, callerencode, calledencode, returnrate, argrate, seed, trlength):

	start_recording = False
	func_sum = []
	func_name = []
	func_return = []
	func_argument = []
	testing_case = []
	testing_case_fc = []
	numbersum = []
	returnsum = []
	argsum = []
	callersum = []
	calledsum = []
	testsum = num_of_tests + num_of_fail
	pass_oneline_num = []   ###Passにおける一行トレースのインデックスを格納
	fail_oneline_num = []   ###Failにおける一行トレースのインデックスを格納

	for i in range(1, testsum + 1):
		callerstring = []
		calledstring = []
		caller_callee_string = []
		argarray = []
		returnarray = []
		numsumtmp = []
		funcstring = []
		funcstring_copy = []
		postorder = []
		trace = []
		arg_return = []
		start_recording = False
		switch = False
		line_no = 1
		if i <= num_of_tests:
			tesutnumber = str(i).zfill(4)
			f = open(trace_path + "/pass_" + tesutnumber + ".csv", 'r')
		else:
			tesutnumber = str(i - num_of_tests).zfill(4)
			f = open(fail_path + "/fail_" + tesutnumber + ".csv", 'r')
		#for index, line in enumerate(f):
		while True:
			line = f.readline()
			if line == "Trace Function\n":
				continue
			if line:
				#try:
					singlecaller = []
					singlecalled = []
					caller = "DUMMY"
					called = "DUMMY"		
					if ABLATION_FC_LINE:
						if caller in TARGET_FC_NAME_LIST and called in TARGET_FC_NAME_LIST:
							#print(caller, called)
							continue
						elif SECOND_TARGET_FLAG and ((caller in SECOND_TARGET_FC_NAME_LIST) or (called in SECOND_TARGET_FC_NAME_LIST)):
							#print(caller, called)
							continue
					
					funcstring.append(line)
			else:
				break
		
		funcstring_copy = copy.deepcopy(funcstring)   ###Deepcopyを作成して書き込み用に編集する

		if SHUFFLE_TR:
			iin = 0
			for fc_index, fc_line in enumerate(funcstring):   ###関数名による行単位でのアブレーション
				caller_s = fc_line
				called_s = fc_line
				#print(caller_s, called_s)
				#print("-------")
				if STATIC_CALL_ENABLE:   ###関数呼び出し関係を考慮する場合
					target_sh_name_caller = TARGET_SH_NAME[0][0]
					target_sh_name_called = TARGET_SH_NAME[0][1]
				else:
					target_sh_name_caller = TARGET_SH_NAME[0]
					#target_sh_name_caller = ["dummyfunction"]   ###呼び出し先関数名に含まれる場合だけアブレーションする場合実行
					target_sh_name_called = TARGET_SH_NAME[0]
					#target_sh_name_called = ["dummyfunction"]   ###呼び出し元関数名に含まれる場合だけアブレーションする場合実行

				if caller_s in target_sh_name_caller or called_s in target_sh_name_called:   ###or ←→ and目的によって使い分ける
				###呼び出し関係を考慮する場合はAND、しない場合はOR(基本OR)
					#print("ii")
					iin += 1
					#print(caller_s, called_s)
					#print(called_s)
					funcstring[fc_index] = "THISISSAKUJO ,"   ###削除対象の要素を特定の文字に入れ替える
					#returnarray[fc_index] = "THISISSAKUJO ,"
					#argarray[fc_index] = "THISISSAKUJO ,"
					funcstring_copy[fc_index] = "THISISSAKUJO ,"

			'''
			funcstring = [p for p in funcstring if p != "THISISSAKUJO ,"]   ###削除対象の要素を削除する
			returnarray = [p for p in returnarray if p != "THISISSAKUJO ,"]
			argarray = [p for p in argarray if p != "THISISSAKUJO ,"]
			funcstring_copy = [p for p in funcstring_copy if p != "THISISSAKUJO ,"]
			'''
		if END_FUNC_ABLATION and len(funcstring) > END_FUNC_NUM * 10:   ###最終行周辺を一括でアブレーション処理する
			for end_line in range(END_FUNC_NUM):
				line_end_number = -(end_line + 2)   ###最終行はAttention Weightの計算に用いるため残留
				#print(funcstring[line_end_number], returnarray[line_end_number], argarray[line_end_number])
				funcstring[line_end_number] = "THISISSAKUJO ,"   ###削除対象の要素を特定の文字に入れ替える
				#returnarray[line_end_number] = "THISISSAKUJO ,"
				#argarray[line_end_number] = "THISISSAKUJO ,"
				funcstring_copy[line_end_number] = "THISISSAKUJO ,"
		if PER_LINE_ABLATION:
			if len(PER_LINE_TEST_CASE_PASS) != 0:
				if i in PER_LINE_TEST_CASE_PASS:
					print("Line Ablation is starting.")
					per_line_index = PER_LINE_TEST_CASE_PASS.index(i)   ###何行目を削除するのかの情報が格納されているリストのインデックスを所得
					if PASS_TRACES_A_to_B_ABLATION:
						for line_no_list in ABLATION_LINE_NO_PASS[per_line_index]:
							for line_index_number in range(line_no_list[0], line_no_list[1] + 1):   ###rangeの仕様上、＋１が必要
								print(funcstring[line_index_number], returnarray[line_index_number], argarray[line_index_number])
								funcstring[line_index_number] = "THISISSAKUJO ,"   ###削除対象の要素を特定の文字に入れ替える
								#returnarray[line_index_number] = "THISISSAKUJO ,"
								#argarray[line_index_number] = "THISISSAKUJO ,"
								funcstring_copy[line_index_number] = "THISISSAKUJO ,"
					else:
						for line_index_number in ABLATION_LINE_NO_PASS[per_line_index]:
							print(funcstring[line_index_number], returnarray[line_index_number], argarray[line_index_number])
							funcstring[line_index_number] = "THISISSAKUJO ,"   ###削除対象の要素を特定の文字に入れ替える
							#returnarray[line_index_number] = "THISISSAKUJO ,"
							#argarray[line_index_number] = "THISISSAKUJO ,"
							funcstring_copy[line_index_number] = "THISISSAKUJO ,"

					if (len(ABLATION_CALL_PASS) != 0):
						for line_index_number in ABLATION_CALL_PASS[per_line_index]:
							funcstring[line_index_number] = "THISVALUEISABLATED."
					print("Line Ablation is completed.")
			if len(PER_LINE_TEST_CASE_FAIL) != 0:
				if i in PER_LINE_TEST_CASE_FAIL:
					print("Line Ablation is starting.")
					per_line_index = PER_LINE_TEST_CASE_FAIL.index(i)   ###何行目を削除するのかの情報が格納されているリストのインデックスを所得
					if FAIL_TRACES_A_to_B_ABLATION:
						for line_no_list in ABLATION_LINE_NO_FAIL[per_line_index]:
							for line_index_number in range(line_no_list[0], line_no_list[1] + 1):   ###rangeの仕様上、＋１が必要
								print(funcstring[line_index_number], returnarray[line_index_number], argarray[line_index_number])
								funcstring[line_index_number] = "THISISSAKUJO ,"   ###削除対象の要素を特定の文字に入れ替える
								#returnarray[line_index_number] = "THISISSAKUJO ,"
								#argarray[line_index_number] = "THISISSAKUJO ,"
								funcstring_copy[line_index_number] = "THISISSAKUJO ,"
					else:
						for line_index_number in ABLATION_LINE_NO_FAIL[per_line_index]:
							print(funcstring[line_index_number], returnarray[line_index_number], argarray[line_index_number])
							funcstring[line_index_number] = "THISISSAKUJO ,"   ###削除対象の要素を特定の文字に入れ替える
							#returnarray[line_index_number] = "THISISSAKUJO ,"
							#argarray[line_index_number] = "THISISSAKUJO ,"
							funcstring_copy[line_index_number] = "THISISSAKUJO ,"

					if (len(ABLATION_CALL_FAIL) != 0):
						for line_index_number in ABLATION_CALL_FAIL[per_line_index]:
							funcstring[line_index_number] = "THISVALUEISABLATED."
					print("Line Ablation is completed.")

		funcstring = [p for p in funcstring if p != "THISISSAKUJO ,"]   ###削除対象の要素を削除する
		#returnarray = [p for p in returnarray if p != "THISISSAKUJO ,"]
		#argarray = [p for p in argarray if p != "THISISSAKUJO ,"]
		funcstring_copy = [p for p in funcstring_copy if p != "THISISSAKUJO ,"]

		if ONE_LINE_TRACE_DELETE:
			if len(funcstring) < DEL_TRACE_LINE_NUM and i <= num_of_tests:   ###x行のトレースを削除する(pass)
				pass_oneline_num.append(i)
				continue
			elif len(funcstring) < DEL_TRACE_LINE_NUM and i > num_of_tests:   ###x行のトレースを削除する(fail)
				fail_oneline_num.append(i - num_of_tests)
				continue
			else:
				func_sum.append(funcstring_copy)
				func_name.append(funcstring)
				#func_return.append(returnarray)
				#func_argument.append(argarray)
		else:
			func_sum.append(funcstring_copy)
			func_name.append(funcstring)
			#func_return.append(returnarray)
			#func_argument.append(argarray)

		if i % 100 == 0:
			print(i, "is complete\n")
	
	if W2VEC_LEARNING:
		print("Start Word2Vec Learning...")
		start = time.time()
		model_name = word2vec.Word2Vec(sentences=func_name, vector_size=W_V, window=WINDOW_, min_count=1, epochs=270, sg=1)  #epoch=270
		print("Function name is encoded.\n")
		#model_return = word2vec.Word2Vec(sentences=func_return, vector_size=W_V_RETURN, window=WINDOW_, min_count=1, epochs=270, sg=1)  #epoch=270
		#print("Return values is encoded.\n")
		#model_argument = word2vec.Word2Vec(sentences=func_argument, vector_size=W_V_ARGUMENT, window=WINDOW_, min_count=1, epochs=270, sg=1)  #epoch=270
		#print("Arguments is encoded.\n")
		word2vec_conv_time = time.time() - start
		print("convoltion time: ", word2vec_conv_time / 60)
		if USE_EXPERIMENT_MODE:
			if os.path.exists(TRACE_VERSION + "/MODEL/" + "{}".format(seed)):
				print("OVERWRITE RISK OCCURED")
				print(TRACE_VERSION + "/MODEL/" + "{}".format(seed) + "  is exists.")
				sys.exit()
			else:
				os.mkdir(TRACE_VERSION + "/MODEL/" + "{}".format(seed))
				model_name.save(TRACE_VERSION + "/MODEL/" + "{}".format(seed) + W2V_MODEL_NAME_NAME)
				#model_return.save(TRACE_VERSION + "/MODEL/" + "{}".format(seed) + W2V_MODEL_NAME_RETURN)
				#model_argument.save(TRACE_VERSION + "/MODEL/" + "{}".format(seed) + W2V_MODEL_NAME_ARGUMENTS)
	else:
		if USE_EXPERIMENT_MODE:
			print(TRACE_VERSION + "/MODEL/" + "{}".format(seed) + W2V_MODEL_NAME_NAME)
			model_name = Word2Vec.load(TRACE_VERSION + "/MODEL/" + "{}".format(seed) + W2V_MODEL_NAME_NAME)
			#model_return = Word2Vec.load(TRACE_VERSION + "/MODEL/" + "{}".format(seed) + W2V_MODEL_NAME_RETURN)
			#model_argument = Word2Vec.load(TRACE_VERSION + "/MODEL/" + "{}".format(seed) + W2V_MODEL_NAME_ARGUMENTS)
		else:
			#model = Word2Vec.load(MODEL_LOAD_PATH)
			print("this mode cannot use.\n")
			sys.exit()
	#documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(object)]
	#model_doc = Doc2Vec(documents, vector_size=256, window=10, min_count=1, workers=4)

	return func_name, func_return, func_argument, model_name, pass_oneline_num, fail_oneline_num#, model_doc, documents

def isfloat(s):  # 浮動小数点数値を表しているかどうかを判定
    try:
        float(s)  # 文字列を実際にfloat関数で変換してみる
    except ValueError:
        return False  # 例外が発生＝変換できないのでFalseを返す
    else:
        return True  # 変換できたのでTrueを返す
def str_to_int(arr):
	returnstr = ""
	arr = arr.split()
	for s in arr:
		#print(s)
		if isfloat(s) == False:
			for w in s:
				returnstr += str(ord(w))
				returnstr += " "
		elif str(s) == "nan":
			returnstr += "0"
			returnstr += " "
		else:
			#s = str(round(float(s), 3))
			returnstr += s
			returnstr += " "
	return returnstr

def process_data(execution_trace_list, model_name, encoding_size, data_oneline_index):   ###すべてのトレースに対してテンソル型に変換する関数
	trace_tensor_list = []
	string_change_time = 0
	#print("lenfuncsum:",len(func_sum))
	#print(func_sum['label'])
	trace_index = 0
	not_oneline_trace_index = 0
	for input_trace_data in execution_trace_list:
		trace_index += 1
		not_oneline_trace_index += 1
		while True:   ###インデックスズレを防ぐ機構
			if not_oneline_trace_index in data_oneline_index:
				not_oneline_trace_index += 1
			else:
				break
		data = {"li": [], "label": 'pass', "data_index": not_oneline_trace_index, "this_data_index": trace_index - 1}   ###this_data_indexはこの関数においてこのトレースが何番目に処理されたかを示すインデックス番号。このデータが複数格納されたリストをシャッフルする際に活用される。
		for vec_index, Li in enumerate(input_trace_data['data_name']):
			if (Li == "THISVALUEISABLATED."):
				Li_vec_name = np.zeros(W_V)   ###zeroベクトルで置換対象
			else:
				if CALL_FULL_ABL:   ###関数名を全てアブレーションする場合
					Li_vec_name = np.zeros(W_V)
				else:
					Li_vec_name = model_name.wv[Li]   ###置換対象でない関数名はWord2Vecのベクトル
			'''
			if (input_trace_data['data_return'][vec_index] == "THISVALUEISABLATED."):
				Li_vec_return = model_return.wv[BLANK_STR]   ###返り値のみのアブレーションでは指定したBLANKのベクトルが使用される
				#Li_vec_return = np.zeros(W_V_RETURN)   ###帰り値のみのアブレーションの場合、ゼロベクトルが代わりに使用される
			else:
				if RETURN_FULL_ABL:   ###返り値のみアブレーションする場合
					Li_vec_return = model_return.wv[BLANK_STR]
				else:
					Li_vec_return = model_return.wv[input_trace_data['data_return'][vec_index]]
				#test_vec = np.zeros(50)
				#print(type(Li_vec_return), Li_vec_return.shape, test_vec.shape)
			if input_trace_data['data_argument'][vec_index] == "THISVALUEISABLATED.":
				Li_vec_argument = model_argument.wv[BLANK_STR]   ###引数のみのアブレーションでは指定したBLANKのベクトルが使用される
				#Li_vec_argument = model_return.wv[BLANK_STR]
				#Li_vec_argument = np.zeros(W_V_ARGUMENT)    ###引数のみのアブレーションの場合、ゼロベクトルが代わりに使用される
			else:
				if ARUGUMENT_FULL_ABL:
					Li_vec_argument = model_argument.wv[BLANK_STR]
				else:
					Li_vec_argument = model_argument.wv[input_trace_data['data_argument'][vec_index]]
			'''
			start = time.time()
			Li_tensor_name = str_to_tensor(Li_vec_name, W_V)
			#Li_tensor_return = str_to_tensor(Li_vec_return, W_V_RETURN)
			#Li_tensor_argument = str_to_tensor(Li_vec_argument, W_V_ARGUMENT)
			string_change_time += time.time() - start
			#print("index:",index)
			#Li_tensor = torch.cat((Li_tensor_name, Li_tensor_return, Li_tensor_argument), 2)
			Li_tensor = Li_tensor_name
			'''
			print(Li_tensor_name)
			print(Li_tensor_return)
			print(Li_tensor_argument)
			print(Li_tensor)
			sys.exit()
			'''
			data['li'].append(Li_tensor)
			#print(Li_tensor.shape)
		if input_trace_data['label'] == 'pass':
			data['label'] = 'pass'
		else:
			data['label'] = 'fail'

		trace_tensor_list.append(data)
		if trace_index % 100 == 0:
			print(trace_index, " is completed.")
		#print(vec_index)
	return trace_tensor_list

def str_to_tensor(str_list, encoding_size):

	if encoding_size == 0:
		tensor = torch.FloatTensor(list(map(float, str_list))).unsqueeze(0).unsqueeze(0)
		tensor = tensor
		tensor = try_gpu(tensor)
		#print(tensor.shape)
		test_tensor(tensor)
		return tensor
	else:
		float_list = list(map(float, str_list))
		#print("str:",len(str_list))
		#print("float:",len(float_list))
		tensor = torch.randn(int(len(float_list) / encoding_size), encoding_size)
		#print(tensor.shape)
		for index, item in enumerate(float_list):
			#print(item)
			tensor[int(index / encoding_size)][index % encoding_size] = item
		tensor = tensor.unsqueeze(1)
		tensor = tensor
		tensor = try_gpu(tensor)
		test_tensor(tensor)
		return tensor

def test_tensor(tensor):
	assert torch.isnan(tensor).any() == 0
	return

def lstm_network(func_sum, encoding_size, data_size, nn_layer):

	string_change_time = 0
	lstm_hidden_layer = []

	lstm_vector = {"data": [], "label": 'pass'}
	#for index, trace in enumerate(data):
	#	lstm_vector.append({'data': [], 'label': 'pass'})
	if BiDIRECT_LSTM:
		h0 = torch.randn(2, 1, W_V * 2)
		c0 = torch.randn(2, 1, W_V * 2)
	else:
		h0 = torch.randn(1, 1, W_V * 2)
		c0 = torch.randn(1, 1, W_V * 2)
	h0 = try_gpu(h0)
	c0 = try_gpu(c0)
	randomtensor = torch.randn(len(func_sum['li']), 1, W_V)
	randomtensor = try_gpu(randomtensor)
	#randomtensor = randomtensor.cuda()
	for ind, d in enumerate(func_sum['li']):
		randomtensor[ind] = d
	'''
	for Li_tensor in data['li']:
		#print(Li_tensor)
		out, (h0, c0) = nn_layer(Li_tensor, (h0, c0))
		#print("lstn:",out[-1].shape)
		test_tensor(out.cuda())
	'''
	if USING_RNN:
		out, h0 = nn_layer(randomtensor, h0)
	else:
		out, (h0, c0) = nn_layer(randomtensor, (h0, c0))
	for hv in out:   ###LSTMの各レイヤーにおける隠れ層の値を格納しておく
		lstm_hidden_layer.append(hv)
		#print(hv.shape)
	lstm_vector['data'] = out[-1]
	lstm_vector['data'] = try_gpu(lstm_vector['data'])
	if func_sum['label'] == 'pass':
		lstm_vector['label'] = 'pass'
	else:
		lstm_vector['label'] = 'fail'
	#print(out[-1].shape)
	#print(len(lstm_vector))
	return lstm_vector, lstm_hidden_layer, string_change_time

def compute_attention_forward(model, lstm_hidden_vector_list):   ###attention機構を用いてLSTMが出力する隠れ層への重み付けを行い、トレースを１つのベクトルにする関数。
	#attention_vector = model['attention'].weight   ###attention機構における重み（アテンションベクトル）
	attention_vector = lstm_hidden_vector_list[-1]
	attention_vector = attention_vector.squeeze()   ###軸が1の次元をなくす([0]と同じ)
	attention_vector = try_gpu(attention_vector)

	attention_vector_dot_hidden_list = []
	for hs in lstm_hidden_vector_list[:-1]:
		hs = try_gpu(hs.squeeze())
		dot_compute_vector = torch.dot(hs, attention_vector)   ###LSTMの各レイヤーにおける隠れ層とアテンションベクトルの内積
		#dot_compute_vector = cos_sim_torch(hs, attention_vector)   ###LSTMの各レイヤーにおける隠れ層とアテンションベクトルのコサイン類似度
		attention_vector_dot_hidden_list.append(dot_compute_vector)

	attention_vector_dot_hidden_tensor = torch.tensor(attention_vector_dot_hidden_list).unsqueeze(0)
	#print(attention_vector_dot_hidden_tensor[0])
	###unsqueezeは次元を1ふやす（shapeは1）

	softmax_attention = nn.Softmax(dim=1)   ###内積した値の加重和を計算するためのsoftmax
	#print(attention_vector_dot_hidden_tensor.shape)
	attention_weight = softmax_attention(attention_vector_dot_hidden_tensor)   ###Attention VeectorとLSTM出力値内積のSoftmaxがAttention重み
	#attention_weight = attention_vector_dot_hidden_tensor    ###Attention VeectorとLSTM出力値内積がAttention重み
	#print(attention_weight)

	zero_list = [0] * LSTM_OUTPUT_SIZE   #LSTMの各レイヤーの隠れ層にアテンションウェイトをかけたものの総和が１つのトレースを表すベクトルになる([1, 256])
	#print(zero_list)
	return_input_vector = torch.tensor(zero_list)
	return_input_vector = try_gpu(return_input_vector)

	for hs_index, hs in enumerate(lstm_hidden_vector_list[:-1]):
		hs = try_gpu(hs)
		#print(hs.shape, attention_weight.shape)
		weight_list = [attention_weight[0][hs_index]] * LSTM_OUTPUT_SIZE   ###[attention weight]が行方向にLSTM hidden sizeの数だけ並ぶ(shape[LSTM_HIDDEN_V])
		weight_tensor = torch.tensor(weight_list)
		weight_tensor = try_gpu(weight_tensor)
		#print(hs.shape, weight_tensor.shape)   ###shape[1, 256], shape[256]
		return_input_vector = torch.add( return_input_vector, hs * weight_tensor )   ###LSTMの隠れ層とアテンションウェイトをかけたものを加算していく

	#print(return_input_vector.shape)

	return return_input_vector, attention_weight

def cos_sim_torch(v1, v2):
	cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
	#print(v1.shape)
	return cos(torch.unsqueeze(v1, 0), torch.unsqueeze(v2, 0))
	#return torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))

def execute_network(input_trace, model):

	input_trace, lstm_hidden_vector_list, string_change_time = lstm_network(input_trace, W_V, len(input_trace), model['line'])
	if ATTENTION_ENABLE:
		attention_input_vector, attention_weight_list = compute_attention_forward(model, lstm_hidden_vector_list)
		scores = mlp_forward(model, attention_input_vector)
		return scores, input_trace, attention_weight_list, string_change_time
	else:
		if MLP_INPUT_IS_AVERAGE:
			###LSTMの各隠れ層の値を平均化する処理
			mlp_input_vector = torch.tensor( [0] * LSTM_HIDEN_V )
			mlp_input_vector = try_gpu(mlp_input_vector)
			for hs_index, hs in enumerate(lstm_hidden_vector_list):
				mlp_input_vector = torch.add( mlp_input_vector, hs )
			mlp_input_vector = torch.div(mlp_input_vector, hs + 1)
			scores = mlp_forward(model, mlp_input_vector)
		else:
			scores = mlp_forward(model, input_trace['data'])
		#print(scores)
		return scores, input_trace, False, string_change_time   ###attentionを利用しない場合は、アテンションウェイトを返す変数にはFalseを返す
		#return input_trace, input_trace

def mlp_forward(model, inp_tensor):

	test_tensor(inp_tensor)

	out = model['mlp'][0](inp_tensor)
	test_tensor(out)
	for i in range(1, len(model['mlp'])):
		out = model['mlp'][i](out)
		test_tensor(out)

	return out

def output_set_labelling(datapoint_set, model):

	p_list = []
	lstm_vector = []
	lstm_v_target_val = np.array([])
	lstm_v_target_fail = np.array([])   #tuikabun
	instrument_time_exe = 0
	total_string_list_change_time = 0

	attention_weight_return = []
	with torch.no_grad():
		pass_match, fail_match, pass_total, fail_total = 0, 0, 0, 0
		for d_index, datapoint in enumerate(datapoint_set):
			if d_index % 50 == 0:
				print(d_index, "is complete.")
			start = time.time()
			d_output, lstm_v, attention_weight_vector, string_change_time = execute_network(datapoint, model)
			instrument_time_exe += time.time() - start
			total_string_list_change_time += string_change_time

			if ATTENTION_ENABLE:
				attention_weight_return.append(attention_weight_vector)
			lstm_vector.append(lstm_v)
			if "pass" in datapoint['label']:
				lstm_v_target_val = np.append(lstm_v_target_val, 0)
				pass_total += 1
				p_list.append("{} 1".format(str(d_output.item())) )
				if d_output.item() >= 0.5:
					pass_match += 1
					print("index: ", d_index, "point: ", d_output.item())
				else:
					print("wrong data number(pass → fail):", datapoint['data_index'], "\npoint:", d_output.item())
			elif "fail" in datapoint['label']:
				#print(d_index)
				lstm_v_target_fail = np.append(lstm_v_target_fail, 1)
				fail_total += 1
				p_list.append("{} 0".format(str(d_output.item())) )
				if d_output.item() <= 0.5:
					fail_match += 1
					print("index: ", d_index, "data_index: ", datapoint['data_index'], "point: ", d_output.item())
				else:
					print("wrong data number(fail → pass):", datapoint['data_index'], "\npoint:", d_output.item())
					###datapoint['data_index']は実行痕跡のファイル名における番号に対応する。ex: datapoint['data_index'] = 10の場合、実行痕跡のファイル名はfunctionname10.logになる。
			else:
				assert False, "Unrecognized label"

		#if ep == 0 or ep == 9 or ep == 19 or ep == 29 or ep == 39:
			#pca_exe(lstm_vector, name, ep)
		'''
		X_pass_val, X_fail_val = convert2d(lstm_vector, ep, name)
		X_pass_val_np, X_fail_val_np = data_get_fromlstm(lstm_vector)
		val_np = np.concatenate([X_pass_val_np, X_fail_val_np])
		Y_val = np.concatenate([lstm_v_target_val, lstm_v_target_fail])
		val = np.concatenate([X_pass_val, X_fail_val])
		'''
		'''
		mglearn.discrete_scatter(val[:, 0], val[:, 1], Y_val)
		plt.legend(['pass', 'fail'], loc='best')
		plt.show()
		'''
	print("instrument_exe_time :", instrument_time_exe / 60)
	print("total_string_list_change_time: ", total_string_list_change_time / 60)

	return pass_match, pass_total, fail_match, fail_total, attention_weight_return, p_list, total_string_list_change_time#, val, Y_val, val_np

def print_out_top_attention_weight(attention_weight_vector_for_searching, input_data, hukusu_get_flag=True):   ###与えられた重みのリストから最大のインデックスを返す関数
	max_point = -99999
	max_index = 99999
	#print(attention_weight_vector)
	for w_index, weight in enumerate(attention_weight_vector_for_searching[0]):
		if weight >= max_point:
			max_index = w_index
			max_point = weight
	if hukusu_get_flag:   ###複数個のインデックスを所得したい場合にTrueで使用。この関数print_out_top_attention_weightを続けて使用することで二番目に大きい要素のインデックスを所得可能
		attention_weight_vector_for_searching[0][max_index] = -9999

	max_Li = input_data['data_name'][max_index]   ###貢献度が高いトレースの行を抽出
	if max_Li.split('\n')[0] != "THISVALUEISABLATED.":
		caller_method = max_Li.split('\n')[0].split(' ')[0]   ###リデューストレースのトレースに対する操作になっていることに注意
		called_method = max_Li.split('\n')[0].split(' ')[1]
	else:
		caller_method = "DELETED"
		called_method = "DELETED"
	#print(caller_method, called_method)
	caller_and_called = caller_method + " "
	caller_and_called += called_method

	return max_index, attention_weight_vector_for_searching, caller_method, called_method, caller_and_called   ###返り値は最も注意重みが大きいトレースのインデックス、その重みの値を0にしたリスト、貢献度が高い呼び出し元関数名、呼び出し先関数名, その二つの組み合わせ

def get_each_trace_attention_weight(attention_weight_vector, input_data):   ###引数として受け取ったトレースにおける、各関数情報のAttention重みの値をリストに順に格納して返す関数
	return_each_function_weight_list = []   ###返り値となるリスト
	for w_ind, w in enumerate(input_data['data_name'][:-1]):   ###Attentionの仕組み上、最後の関数呼び出し情報は対象にならない。
		each_function_attention_weight = attention_weight_vector[0][w_ind]    ###各関数呼び出し情報のAttenion重み
		return_each_function_weight_list.append(each_function_attention_weight)

	return return_each_function_weight_list

def check_and_add_value_to_dic(aggregation_method_attention_dic, add_target_list):   ###注意重みが大きい関数名を集計するためのディクショナリを操作する関数
	for target_method_name in add_target_list:
		if target_method_name in aggregation_method_attention_dic:
			aggregation_method_attention_dic[target_method_name] += 1   ###すでにキーが登録済みの場合はその関数名のカウントをプラス1する
		else:
			aggregation_method_attention_dic[target_method_name] = 1   ###キーが登録済みでない場合は新たに登録する
	return aggregation_method_attention_dic

def write_attention_weight_each_traces(attention_weight_write_list, write_dir_path, sakujo_trace_index, testcase_num):

	sakujo_count = 0
	#print(len(attention_weight_write_list), len(sakujo_trace_index), testcase_num)
	for current_write_index in range(testcase_num):
		write_dir_path_open = write_dir_path + "trace{}.log".format(current_write_index + 1)
		weiw = open(write_dir_path_open, "w")
		if current_write_index + 1 in sakujo_trace_index:
			weiw.write("This trace is deleted.\n")
			sakujo_count += 1
		else:
			tr_weight_l = attention_weight_write_list[current_write_index - sakujo_count]
			for tr_ind, tr_weight in enumerate(tr_weight_l):   ###一つのトレースにおけるAttention重みのリストの繰り返し
				str_weight = str(tr_ind + 1) + " : " + str(tr_weight) + "\n"
				weiw.write(str_weight)
		weiw.close()
	return 0

def make_trainingset(training_length, data_pass, data_fail):
	#トレーニングセットと評価用データセットの作成とデータ数の調整
	setting_num = int(training_length * (len(data_pass) + len(data_fail)))   #トレーニングデータの数
	training_set = data_pass[:int(training_length*len(data_pass))] + data_fail[:int(training_length*len(data_fail))]   #こっちが通常のサンプリング
	validation_set = data_pass[int(training_length*len(data_pass)):] + data_fail[int(training_length*len(data_fail)):]   #こっちが通常のサンプリング
	#training_set = data_pass[:int(training_length*len(data_pass))] + data_fail[int((1 - training_length)*len(data_fail)):]
	'''   #データ選択をマニュアルで選定する場合   データ番号指定。インデックス指定ではない
	index_list_pass = [2, 4, 15, 21, 25, 55]
	index_list_fail = [2, 11]
	training_set, validation_set = make_training_set(data_pass, data_fail, index_list_pass, index_list_fail)   ###todo: valition setのプログラム→complete 2021/11/24
	'''
	print("training pass", len(data_pass[:int(training_length*len(data_pass))]), "training fail:", len(data_fail[:int(training_length*len(data_fail))]), "val pass:", len(data_pass[int(training_length*len(data_pass)):]), "val fail:", len(data_fail[int(training_length*len(data_fail)):]))

	if len(training_set) < setting_num:   #トレーニングデータの数が1足りない場合
		if int(training_length*len(data_pass)) < int(training_length*len(data_fail)):   #failが多い場合はpassを1増やす
			training_set = data_pass[:int(training_length*len(data_pass)) + 1] + data_fail[:int(training_length*len(data_fail))]
			validation_set = data_pass[int(training_length*len(data_pass)) + 1:] + data_fail[int(training_length*len(data_fail)):]
			print("training pass", len(data_pass[:int(training_length*len(data_pass)) + 1]), "training fail:", len(data_fail[:int(training_length*len(data_fail))]), "val pass:", len(data_pass[int(training_length*len(data_pass)) + 1:]), "val fail:", len(data_fail[int(training_length*len(data_fail)):]))

		else:   #passが多い場合はfailを1増やす
			training_set = data_pass[:int(training_length*len(data_pass))] + data_fail[:int(training_length*len(data_fail)) + 1]
			validation_set = data_pass[int(training_length*len(data_pass)):] + data_fail[int(training_length*len(data_fail)) + 1:]
			print("training pass", len(data_pass[:int(training_length*len(data_pass))]), "training fail:", len(data_fail[:int(training_length*len(data_fail)) + 1]), "val pass:", len(data_pass[int(training_length*len(data_pass)):]), "val fail:", len(data_fail[int(training_length*len(data_fail)) + 1:]))
	return training_set, validation_set

def classification(data_pass, data_fail, inference_data, model, s, trace_str_list, passnum, failnum, pass_oneline_num, fail_oneline_num):

	random.seed(100)
	tr_only_classify = []   ###訓練データのみ分類する際に使用する分類するトレースのデータインデックスを格納するリスト
	tr_only_pass_ind = []
	tr_only_fail_ind = []

	output_list = []

	if TRAINIG_DATA_ONLY_CALSSIFY:  ###訓練データのみ予測分類する場合
		for i in range(s):
			#shuffle(inference_data)
			shuffle(data_pass)
			shuffle(data_fail)
		training_set, validation_set = make_trainingset(TRAINING_SIZE_, data_pass, data_fail)
		for inf_data in training_set:
			tr_only_classify.append(inf_data['data_index'])
			if inf_data['label'] == 'pass':
				tr_only_pass_ind.append(inf_data['data_index'])
			else:
				tr_only_fail_ind.append(inf_data['data_index'])
		print("This execution classifies only training data set.\n", "FULL DATA SIZE: ", passnum + failnum, "\nCLASSIFIED DATA SIZE:", len(training_set))
	#print(training_set[0])

	#'''
	aggregation_method_attention_dic = {}
	aggregation_caller_and_called_dic = {}   ###呼び出し元と呼び出し先の組み合わせ集計用
	attention_weight_each_traces_list = []   ###各トレースにおけるアテンション重みの値のリストを格納するリスト

	with torch.no_grad():
		for index, in_data in enumerate(inference_data):
			if TRAINIG_DATA_ONLY_CALSSIFY:
				if in_data['label'] == 'pass':   ###Attention Weightを書き込む際に書き込みファイルのズレをなくすための措置。分類しないトレースは一行のトレースとして扱うことで書き込み処理から除外される。
					if in_data['data_index'] not in tr_only_pass_ind:
						pass_oneline_num.append(in_data['data_index'])
						attention_weight_each_traces_list.append([0])   ###Attention Weight書き込み用の処理
						continue   ###訓練データ以外は分類予測しない
				else:
					if in_data['data_index'] not in tr_only_fail_ind:
						fail_oneline_num.append(in_data['data_index'])
						attention_weight_each_traces_list.append([0])   ###Attention Weight書き込み用の処理
						continue
			each_trace_attention_weight = []   ###一つのトレースにおけるアテンション重みの値を格納するリスト。これをattention_weight_each_traces_listに格納する。

			out, lstm_v, attention_weight_vector, string_change_time = execute_network(in_data, model)
			output_list.append(out)
			attention_weight_vector_for_searching = copy.deepcopy(attention_weight_vector)   ###最もAttention重みが大きい関数情報を探すために使う、オリジナルのコピー
			#print(type(attention_weight_vector))
			if ATTENTION_ENABLE and OUTPUT_ATTENSION_WEIGHT:   ###アテンション機構が有効の時のみ、アテンションウェイトを出力する
				#print(attention_weight_vector[0])
				weight_max_index, attention_weight_vector_2, caller_1, called_1, caller_and_called_1 = print_out_top_attention_weight(attention_weight_vector_for_searching, trace_str_list[index], hukusu_get_flag=True)
				each_trace_attention_weight = get_each_trace_attention_weight(attention_weight_vector, trace_str_list[index])
				attention_weight_each_traces_list.append(each_trace_attention_weight)
				#weight_second_index, attention_weight_vector_3, caller_2, called_2, caller_and_called_2 = print_out_top_attention_weight(attention_weight_vector_2, trace_str_list[index], hukusu_get_flag=True)
				#weight_third_index, attention_weight_vector_4, caller_3, called_3, caller_and_called_3 = print_out_top_attention_weight(attention_weight_vector_3, in_data, hukusu_get_flag=True)
				###三つ目の大きさまでのウェイトを出力する。コメントアウトor行追加で数は変更して。
				#print("Index: ", index, "length: ", len(attention_weight_vector[0]), "\nWeight Ranking:", weight_max_index, weight_second_index, weight_third_index)
				aggregation_method_attention_dic = check_and_add_value_to_dic(aggregation_method_attention_dic, [called_1])#, caller_1])#, caller_2, called_2])#, caller_3, called_3])
				aggregation_caller_and_called_dic = check_and_add_value_to_dic(aggregation_caller_and_called_dic, [caller_and_called_1])#, caller_and_called_2])#, caller_and_called_3])

			if "pass" in in_data['label']:
				target = torch.tensor([[1.]])
				target = try_gpu(target)
			elif "fail" in in_data['label']:
				target = torch.tensor([[0.]])
				target = try_gpu(target)
			else:
				assert False, "Unrecognized label"
	#'''

	###各トレースにおけるAttention重みをファイルに出力する。
	if not os.path.exists(EACH_ATTENTION_WEIGHT_PASS_PATH):   ###ディレクトリの存在確認（ない場合は新たに作成）
		os.mkdir(EACH_ATTENTION_WEIGHT_PASS_PATH)
	else:
		print("Attention weight-pass OVERWRITE.")

	if not os.path.exists(EACH_ATTENTION_WEIGHT_FAIL_PATH):   ###ディレクトリの存在確認（ない場合は新たに作成）
		os.mkdir(EACH_ATTENTION_WEIGHT_FAIL_PATH)
	else:
		print("Attention weight-fail OVERWRITE.")

	if PASS_TRACE_CLASSIFY and OUTPUT_ATTENSION_WEIGHT:
		write_attention_weight_each_traces(attention_weight_each_traces_list[:passnum], EACH_ATTENTION_WEIGHT_PASS_PATH, pass_oneline_num, PASS_TR_NUM)   ###passの書き込みのみ
	elif FAIL_TRACE_CALSSIFY and OUTPUT_ATTENSION_WEIGHT:
		write_attention_weight_each_traces(attention_weight_each_traces_list[0:], EACH_ATTENTION_WEIGHT_FAIL_PATH, fail_oneline_num, FAIL_TR_NUM)   ###failの書き込みのみ
		###FAIL_TRACE_CALSSIFY and OUTPUT_ATTENSION_WEIGHTでPASS_TR_NUM=0の場合、Attentionスコアの書き込みと出力が不可になりプログラムが異常終了する。要修正。
	elif OUTPUT_ATTENSION_WEIGHT:
		write_attention_weight_each_traces(attention_weight_each_traces_list[:passnum], EACH_ATTENTION_WEIGHT_PASS_PATH, pass_oneline_num, PASS_TR_NUM)   ###passの書き込み
		write_attention_weight_each_traces(attention_weight_each_traces_list[passnum:], EACH_ATTENTION_WEIGHT_FAIL_PATH, fail_oneline_num, FAIL_TR_NUM)   ###failの書き込み
	else:
		pass

	start = time.time()
	if TRAINIG_DATA_ONLY_CALSSIFY:
		pass_match, pass_total, fail_match, fail_total, attention_weight_list, _, total_string_list_change_time = output_set_labelling(training_set, model)
	else:
		pass_match, pass_total, fail_match, fail_total, attention_weight_list, _, total_string_list_change_time = output_set_labelling(inference_data, model)
	inference_time_val = time.time() - start
	if WRITE_TIME_INFORMATION:   ###実行時間の書き込み
		timewrite.write_time_imformation(inference_time_val, TRACE_VERSION)

	'''
	print("Attention_each_traces_list_len:", len(attention_weight_each_traces_list))
	for i_ind, i in enumerate(attention_weight_each_traces_list[752]):
		print(i_ind + 1,":", i)
	'''

	if not PASS_TRACE_CLASSIFY:
		recall_a = fail_match / fail_total
	else:
		recall_a = 0
	if not FAIL_TRACE_CALSSIFY:
		TNR_a = pass_match / pass_total
	else:
		TNR_a = 0

	if (fail_match + (pass_total - pass_match)) != 0:
		precision_a = fail_match / (fail_match + (pass_total - pass_match))
	else:
		precision_a = 0
	if recall_a != 0 or precision_a != 0:
		f_measure = 2 * recall_a * precision_a / (recall_a + precision_a)
	else:
		f_measure = 0
	print("\nPrecision: {}\nRecall: {}\nTNR: {}\nF-measure:{}\n".format(precision_a, recall_a, TNR_a, f_measure))

	print("\nClassification set:\npass matches: {}\npass total: {}\nfail matches: {}\nfail total: {}\n".format(pass_match, pass_total, fail_match, fail_total))
	print("inference val time:",inference_time_val / 60)
	print("Inference time (probably): ", (inference_time_val - total_string_list_change_time) / 60)

	aggregation_method_attention_dic = sorted(aggregation_method_attention_dic.items(), key=lambda x:x[1], reverse=True)
	aggregation_caller_and_called_dic = sorted(aggregation_caller_and_called_dic.items(), key=lambda x:x[1], reverse=True)
	print(aggregation_method_attention_dic)
	print(aggregation_caller_and_called_dic)

	if WRITE_OUTPUT_FILE:
		output_class_file = open(OUTPUT_CLASSIFI, "a")
		output_class_file.write("\n---------------------------")
		output_class_file.write("Pop: " + str(POP_KAISU) +  "\nFC_FLAG:" + str(ABLATION_FC_LINE) + "\nTarget_fc_list:\n" + str(TARGET_FC_NAME_LIST) + "\nTR_SHUFFLE_TR:" + str(SHUFFLE_TR) + "\nTarget_list: " + str(TARGET_SH_NAME) + "\nSECOND_TARGET_FLAG:" + str(SECOND_TARGET_FLAG) + "\nSECOND_SH_NAME" + str(SECOND_TARGET_FC_NAME_LIST) + "\nGLOBAL_READ_FROM_FILE: " + str(GLOBAL_READ_FROM_FILE) + "\nG_TARGET_FUNC: " + str(G_TARGET_FUNC) + "\nFront:" + str(FRONT_SAKUJO) + "\n" + PYMODEL_PATH + "\nWord2Vec Path: " + TRACE_VERSION + "/MODEL/" + "{}".format(s) + W2V_MODEL_NAME_NAME + " " + W2V_MODEL_NAME_RETURN + " " + W2V_MODEL_NAME_ARGUMENTS + "\n" + "TRAINIG_DATA_ONLY_CALSSIFY: " + str(TRAINIG_DATA_ONLY_CALSSIFY) + "\nTRAINING_ONLY_SIZE: " + str(TRAINING_SIZE_) + "\n")
		output_class_file.write("\nClassification set:\npass matches: {}\npass total: {}\nfail matches: {}\nfail total: {}\n".format(pass_match, pass_total, fail_match, fail_total))
		output_class_file.write("Precision: " +  str(precision_a) + "\nRecall: " +  str(recall_a) +  "\nTNR: " + str(TNR_a) +  "\n")
		output_class_file.close()

	output_tensor = torch.Tensor(output_list)
	out_average = torch.mean(output_tensor)
	return out_average

def edit_func_sum(data_name, label, data_oneline_index):
	edit_data = []
	not_oneline_index = 1
	print(data_oneline_index)
	for index, seq in enumerate(data_name):
		while True:   ###インデックスデータのズレを訂正するためのループ
			if not_oneline_index in data_oneline_index:
				not_oneline_index += 1
			else:
				break
		edit_data.append({'data_name': [], 'label': 'fail', 'data_index': not_oneline_index})
		for li in seq:
			edit_data[index]['data_name'].append(li)
		if label == "pass":
			edit_data[index]["label"] = "pass"
		else:
			edit_data[index]["label"] = "fail"

	return edit_data

def load_model(load_path, lstm_output_size_a):
	out_feature = [256, 128, 64, 1]
	if USING_RNN:
		if BiDIRECT_LSTM:
			nn_layer = nn.RNN(input_size = W_V,
								hidden_size = 256,
								num_layers = 1,
								bias = False,
								batch_first = False,
								dropout = 0,
								bidirectional = BiDIRECT_LSTM)
		else:
			nn_layer = nn.RNN(input_size = W_V,
								hidden_size = LSTM_HIDEN_V,
								num_layers = 1,
								bias = False,
								batch_first = False,
								dropout = 0,
								bidirectional = BiDIRECT_LSTM)
	else:
		if BiDIRECT_LSTM:
			nn_layer = nn.LSTM(input_size = W_V,
								hidden_size = 256,
								num_layers = 1,
								bias = False,
								batch_first = False,
								dropout = 0,
								bidirectional = BiDIRECT_LSTM)
		else:
			nn_layer = nn.LSTM(input_size = W_V,
								hidden_size = LSTM_HIDEN_V,
								num_layers = 1,
								bias = False,
								batch_first = False,
								dropout = 0,
								bidirectional = BiDIRECT_LSTM)

	if USING_CPU_MACHINE:
		nn_layer.load_state_dict(torch.load(load_path + "line.pth", map_location=torch.device('cpu')))
	else:
		nn_layer.load_state_dict(torch.load(load_path + "line.pth"))
	#print(nn_layer.state_dict())
	nn_layer = try_gpu(nn_layer)
	nn_layer.eval()
	if ATTENTION_ENABLE:
		#attention_layer = nn.Linear(in_features = lstm_output_size_a,
		#							out_features = 1,
		#							bias = False)
		#attention_layer.load_state_dict(torch.load(load_path + "attention.pth"))
		#model = {'line': nn_layer, 'mlp': [], "attention": attention_layer}
		model = {'line': nn_layer, 'mlp': []}
	else:
		model = {'line': nn_layer, 'mlp': []}
	mlp_linear = nn.Linear(in_features = lstm_output_size_a,
									out_features = out_feature[0],
									bias = False)
	if USING_CPU_MACHINE:
		mlp_linear.load_state_dict(torch.load(load_path + f"mlp_0.pth", map_location=torch.device('cpu')))
	else:
		mlp_linear.load_state_dict(torch.load(load_path + f"mlp_0.pth"))
	mlp_linear = try_gpu(mlp_linear)
	mlp_linear.eval()
	#print(mlp_linear.state_dict())
	model["mlp"].append(mlp_linear)
	for i in range(1, len(out_feature)):
		mlp_linear = nn.Linear(in_features = out_feature[i - 1],
										out_features = out_feature[i],
										bias = False)
		if USING_CPU_MACHINE:
			mlp_linear.load_state_dict(torch.load(load_path + f"mlp_{i}.pth", map_location=torch.device('cpu')))
		else:
			mlp_linear.load_state_dict(torch.load(load_path + f"mlp_{i}.pth"))
		mlp_linear = try_gpu(mlp_linear)
		mlp_linear.eval()
		model["mlp"].append(mlp_linear)
		#print(mlp_linear.state_dict())
	#model = {'line': nn_layer}
	return model

def get_fail_line_list():
	fail_line_list = []
	failn = 0
	for coverage_file_path in glob.glob(os.path.join(TRACE_VERSION + "/", "*.csv")):
		if "fail" in coverage_file_path:
			failn += 1
			tesutnumber = str(failn).zfill(4)
			f = open(TRACE_VERSION + "/fail_" + tesutnumber + ".csv", 'r')
			while True:
				line = f.readline()
				if line:
					fail_line_list.append(line)
				else:
					break
	fail_line_list = list(set(fail_line_list))   ###重複の削除
	#print(fail_line_list)
	return fail_line_list

def cal_topn_rank(probscore_list):
	hit_flag = False
	hitted_list = []
	lineofcode = caltopn.get_line_of_code(TRACE_VERSION + "/is_plane.txt")
	line_of_bug, num_of_bug = caltopn.get_line_of_bug("../chunks/" + "{}_{}_buggy_chunks.yaml".format(PROJECT.lower(), VERSION_PROJECT), TRACE_VERSION + "/is_plane.txt")
	#print("ooo\n\n\n\n\n\n\nooo",line_of_bug, num_of_bug, probscore_list)
	rank = 0
	topn_percentile = 100
	for score in probscore_list:
		rank += 1
		#line_of_bug_copy = copy.deepcopy(line_of_bug)
		for lind, linebug in enumerate(line_of_bug):
			if score[0] in linebug:
				if lind not in hitted_list:
					print(score[0])
					num_of_bug -= 1
					hitted_list.append(lind)
			if num_of_bug == 0:
				topn_percentile = 100 * rank / lineofcode
				hit_flag = True
				break
		if hit_flag:
			break
	print("Rank:", rank, "LOC:", lineofcode, "line_of_bug", line_of_bug)
	return topn_percentile

def main(s):

	passnum = PASS_TR_NUM
	failnum = FAIL_TR_NUM
	training_size = 0.3
	returnrate = 0
	argrate = 0
	func_name, func_return, func_argument, model_name, pass_oneline_num_r, fail_oneline_num_r = pool_trace(PASS_TRACES, REDUCED_PASS_TRACES, passnum, FAIL_TRACES, REDUCED_FAIL_TRACES, failnum, returnencode=False, argencode=False, callerencode=False, calledencode=False, returnrate=returnrate, argrate=argrate, seed=s, trlength=training_size)
	#func_sum_fail, model_fail, model_doc_fail, doc_fail = pool_trace(FAIL_TRACES, REDUCED_FAIL_TRACES, 10)
	if len(pass_oneline_num_r) > 0:
		func_sum_pass_total = func_name[:(passnum - len(pass_oneline_num_r))]
		func_sum_fail_total = func_name[(passnum - len(pass_oneline_num_r)):]
		'''
		func_sum_pass_name = func_name[:(passnum - pass_oneline_num_r)]
		func_sum_pass_return = func_return[:(passnum - pass_oneline_num_r)]
		func_sum_pass_argument = func_argument[:(passnum - pass_oneline_num_r)]
		func_sum_fail_name = func_name[(passnum - pass_oneline_num_r):]
		func_sum_fail_return = func_return[(passnum - pass_oneline_num_r):]
		func_sum_fail_argument = func_argument[(passnum - pass_oneline_num_r):]
		'''
	else:
		func_sum_pass_total = func_name[:passnum]
		func_sum_fail_total = func_name[passnum:]
		'''
		func_sum_pass_name = func_name[:passnum]
		func_sum_pass_return = func_return[:passnum]
		func_sum_pass_argument = func_argument[:passnum]
		func_sum_fail_name = func_name[passnum:]
		func_sum_fail_return = func_return[passnum:]
		func_sum_fail_argument = func_argument[passnum:]
		'''
	#'''
	func_sum_pass = edit_func_sum(func_sum_pass_total, "pass", pass_oneline_num_r)
	func_sum_fail = edit_func_sum(func_sum_fail_total, "fail", fail_oneline_num_r)
	#network_model = make_model(256)
	network_model = load_model(PYMODEL_PATH, LSTM_OUTPUT_SIZE)
	print(network_model)
	print("Preprocess Execution Trace Data...")
	if FAIL_TRACE_CALSSIFY:
		func_sum_fail_processed = process_data(func_sum_fail, model_name, W_V, fail_oneline_num_r)
		output_score = classification([], [], func_sum_fail_processed, network_model, s, func_sum_fail, passnum, failnum, pass_oneline_num_r, fail_oneline_num_r)   ###引数の箇所は要改善(訓練データのみ分類機能のための措置)
	elif PASS_TRACE_CLASSIFY:
		func_sum_pass_processed = process_data(func_sum_pass, model_name, W_V, pass_oneline_num_r)
		output_score = classification([], [], func_sum_pass_processed, network_model, s, func_sum_pass, passnum, failnum, passnum, failnum, pass_oneline_num_r, fail_oneline_num_r)
	else:
		func_sum_pass_processed = process_data(func_sum_pass, model_name, W_V, pass_oneline_num_r)
		func_sum_fail_processed = process_data(func_sum_fail, model_name, W_V, fail_oneline_num_r)
		all_data = func_sum_pass_processed + func_sum_fail_processed
		func_sum_pass_dp = copy.deepcopy(func_sum_pass_processed)
		func_sum_fail_dp = copy.deepcopy(func_sum_fail_processed)
		output_score = classification(func_sum_pass_dp, func_sum_fail_dp, all_data, network_model, s, func_sum_pass + func_sum_fail, passnum, failnum, pass_oneline_num_r, fail_oneline_num_r)
	return output_score

TopN_Percentile_list = []   ###SeedごとのTopn％を格納するリスト
#'''
if __name__ == "__main__":   #monte carlo cross validtion
	# seedlist = [i for i in range(20, 30, 1)]
	seedlist = [SEED_VALUE]
	for s in seedlist:
		if USE_EXPERIMENT_MODE:
			elpased_start = time.time()
			if SIMPLE_ABLATION_EXPERIMENT_MODE:
				for ab_target in ABLATION_TARGET_LIST_ZIKKEN:
					TARGET_SH_NAME = [ab_target]
					PYMODEL_PATH = TRACE_VERSION + "/MODEL/" + "{}".format(s) + NNs_MODEL_NAME
					OUTPUT_CLASSIFI = TRACE_VERSION + "/MODEL/" + "{}".format(s) + CLASSIFI_LOG_NAME
					main(s)
			elif LINE_GRANU_ABLATION_MODE:
				ABLATION_TARGET_LIST_ZIKKEN = get_fail_line_list()
				prob_per_statement = dict()
				for ab_ind, ab_target in enumerate(ABLATION_TARGET_LIST_ZIKKEN):
					print("completed ", ab_ind + 1, "/ ", len(ABLATION_TARGET_LIST_ZIKKEN))
					TARGET_SH_NAME = [ab_target]
					PYMODEL_PATH = TRACE_VERSION + "/MODEL/" + "{}".format(s) + NNs_MODEL_NAME
					OUTPUT_CLASSIFI = TRACE_VERSION + "/MODEL/" + "{}".format(s) + CLASSIFI_LOG_NAME
					prob_score = main(s)
					prob_per_statement[ab_target] = float(prob_score)
					#prob_per_statement[ab_ind] = float(prob_score)
				prob_per_statement_copy = copy.deepcopy(prob_per_statement)
				prob_per_statement = sorted(prob_per_statement.items(), key=lambda x:x[1], reverse=True)
				#print(prob_per_statement)
				wfprob = open(TRACE_VERSION + "/MODEL/" + str(s) + "/each_prob.log", "w")
				for state in prob_per_statement:
					wfprob.write(state[0])
					wfprob.write(str(state[1]) + "\n")
				wfprob.close()
				topn_v = cal_topn_rank(prob_per_statement)
				TopN_Percentile_list.append(topn_v)
				print("TopN %: ", topn_v)
				#'''  ###通常はこちら
				wftopn = open(OUTPUT_PER_SEED, "a")
				wftopn.write(TRACE_VERSION + " " + str(topn_v) + "\n")
				wftopn.close()
				#'''
				'''   ###各VersionのProbをLOCを所得して書き込む
				locget = open(TRACE_VERSION + "/is_plane.txt", "r")
				loc_cnt = 0
				loc_inc_comment = 0
				while True:
					forloc = locget.readline()
					if forloc:
						loc_inc_comment += 1
						if forloc != "COMMENT\n":
							loc_cnt += 1
					else:
						break
				locget.close()
				loc_prob_dict = dict()
				locget = open(TRACE_VERSION + "/is_plane.txt", "r")
				tmp_now_ind = 0
				for i in range(loc_inc_comment):
					forloc = locget.readline()
					if forloc != "COMMENT\n":
						loc_prob_dict[tmp_now_ind] = 0
						if forloc in prob_per_statement_copy:
							print("--OK--", tmp_now_ind + 1)
							loc_prob_dict[tmp_now_ind] = prob_per_statement_copy[forloc]
						tmp_now_ind += 1
					else:
						pass
				locget.close()
				probwf = open("seq_sus_ab/" + PROJECT + "_" + VERSION_PROJECT + ".log", "w")
				for lineind, lp in loc_prob_dict.items():
					probwf.write(str(lineind) + ":" + str(lp) + "\n")
				probwf.close()
				'''
			elapsed_time = time.time() - elpased_start
			min = elapsed_time / 60
			sec = elapsed_time % 60
			print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
			print ("elapsed_time:",min,"[min]",sec,"[sec]")
		else:
			elpased_start = time.time()
			main(s)
			elapsed_time = time.time() - elpased_start
			min = elapsed_time / 60
			sec = elapsed_time % 60
			print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
			print ("elapsed_time:",min,"[min]",sec,"[sec]")
	
	if LINE_GRANU_ABLATION_MODE:  ###各SeedのTopN%の値の平均を算出し、ファイルに書き込む
		#pass
		#'''
		average_topN = sum(TopN_Percentile_list) / len(TopN_Percentile_list)
		###Todo書き込み処理の追加
		topn_f = open(WRITE_TOPN_PERCENTILE, "a")
		topn_f.write(PROJECT + " " + VERSION_PROJECT + " " + str(average_topN) + "\n")
		topn_f.close()
		#'''
