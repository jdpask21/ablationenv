from gensim.models import word2vec
from gensim.models.word2vec import Word2Vec
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import torch.nn as nn
import torch.optim as optim
import torch
from random import shuffle
import random
import time
import numpy as np
from gensim.matutils import unitvec
import datetime
import os
import sys
import copy
from ablationex_util import time_information_write as timewrite
import glob
import argparse

parser = argparse.ArgumentParser(description='online classification execution traces.')
parser.add_argument('project_id', help='project ID of defects4j')
parser.add_argument('bug_id', help='bug ID of defects4j')
args = parser.parse_args()

######################################operation val######################################

TRACE_PATH = "../clover-line/"
PROJECT = args.project_id
PROJECT_VERSION = args.bug_id
TRACE_VERSION =  TRACE_PATH + PROJECT + "/" + PROJECT_VERSION
ATTENTION_ENABLE = True
USING_RNN = False   ###Trueで古典的RNNを使用、FalseでLSTMを使用
BiDIRECT_LSTM = False
if BiDIRECT_LSTM:
	LSTM_OUTPUT_SIZE = 512
else:
	LSTM_OUTPUT_SIZE = 256

MLP_INPUT_IS_AVERAGE = True

USE_ZIKKEN_MODE = True

#'''
FAIL_TRACES = TRACE_VERSION
REDUCED_FAIL_TRACES = TRACE_VERSION + "/reduced_traces/fail/"
PASS_TRACES = TRACE_VERSION
REDUCED_PASS_TRACES = TRACE_VERSION + "/reduced_traces/pass/"
#'''
'''   ###schedule用
FAIL_TRACES = TRACE_VERSION + "/ex_traces/fail-not3l/"
PASS_TRACES = TRACE_VERSION + "/ex_traces/pass-not3l/"
'''
'''
FAIL_TRACES = TRACE_VERSION + "/traces/fail/"
REDUCED_FAIL_TRACES = TRACE_VERSION + "/reduced_traces/fail/"
PASS_TRACES = TRACE_VERSION + "/traces/pass/"
REDUCED_PASS_TRACES = TRACE_VERSION + "/reduced_traces/pass/"
'''
EPOCH = "/ep270_ex"
MODEL_EPOCH = "/model_ep12/"
#EPOCH_NUMBER = 150

W2VEC_LEARNING = True   #Trueで学習、Falseで既存モデルをロードする?
W2VEC_FULL_ENCODING = True   ###Trueのみ使用可能
WINDOW_ = 1
W_V_NAME = 28   #単語ベクトルのサイズ   ###W2VEC_FULL_ENCODINGがTrueの場合はここのサイズがFULL encodingのサイズとなる。また、model_nameがfull encodingするWord2Vecのモデルの変数となる。
W_V_RETURN = 50
W_V_ARGUMENT = 50
W2V_MODEL_NAME_NAME = "/wv" + str(W_V_NAME) + "-w" + str(WINDOW_) + ".model"   ###各変数におけるWord2Vecモデルのファイル名（書き込み、読み込み共用）

MODEL_SAVE_PATH = TRACE_VERSION + EPOCH + W2V_MODEL_NAME_NAME
MODEL_LOAD_PATH = TRACE_VERSION + EPOCH + W2V_MODEL_NAME_NAME

SAVE_PYMODEL_FLAG = True   ###trueでpymodelを保存、一定以上性能のみ
SAVE_POINT = 0.8
###MODEL param operation
LEARNING_RATE_ = 0.000001   ##0.0000005, 0.000001(clover-line)
TRAINING_SIZE_ = 0.3
PYMODEL_PATH = TRACE_VERSION + EPOCH  ###pymodelをsaveする場所
TRAIN_LOG_NAME = "/0214-"+ str(int(TRAINING_SIZE_ * 100)) + "w" + str(WINDOW_) + ".log"
TRAIN_LOG_PATH = TRACE_VERSION + EPOCH + TRAIN_LOG_NAME
SEED_VALUE = 45

WRITE_TIME_INFORMATION = True   ###学習時間の計測の際にTrue（書き込みファイルは別ファイルで指定）

###########################################
###########################################  
W_V = W_V_NAME + W_V_RETURN + W_V_ARGUMENT   ###Word2Vecが出力するトータルのベクトルサイズ。三変数分割するためその合計になる。
LSTM_HIDEN_V = W_V * 2   #LSTM隠れ層のサイズ
###########################################
###########################################

TRAINING_FULL_DATA = True  ###全てのデータを訓練データとして扱う場合にTrue

PASS_SEP = 1000   ###DATA_SEP_SET=Trueの場合に使用する変数
FAIL_SEP = 105

ONE_LINE_TRACE_DELETE = True   ###Trueで一行トレースを削除する
DEL_TRACE_LINE_NUM = 2   ###削除するトレースの行数設定。この値以下のトレースは削除される


def get_trace_num():
	failn, passn = 0, 0
	for coverage_file_path in glob.glob(os.path.join(TRACE_VERSION + "/", "*.csv")):
		if "fail" in coverage_file_path:
			failn += 1
		elif "pass" in coverage_file_path:
			passn += 1
	return failn, passn

FAIL_TR_NUM, PASS_TR_NUM = get_trace_num()
EPOCH_NUMBER = int(40 * 200 / (PASS_TR_NUM + FAIL_TR_NUM))   ###エポックはデータ数に依存して決定する
EPOCH_NUMBER = 2
#PASS_TR_NUM = 10
#FAIL_TR_NUM = 10
print(FAIL_TR_NUM, PASS_TR_NUM)

def try_gpu(e):
	if torch.cuda.is_available():
		return e.cuda()
		#return e
	return e

print(PASS_TRACES)
print(FAIL_TRACES)
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
	pass_oneline_num = 0   ###Passにおける一行トレースの数を格納
	fail_oneline_num = 0   ###Failにおける一行トレースの数を格納

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
		while True:
			line = f.readline()
			if line == "Trace Function\n":
				continue
			if line:
				funcstring.append(line)
			else:
				break

		funcstring_copy = copy.deepcopy(funcstring)   ###Deepcopyを作成して書き込み用に編集する

		if ONE_LINE_TRACE_DELETE:
			if len(funcstring) < DEL_TRACE_LINE_NUM and i <= num_of_tests:   ###x行以下のトレースを削除する(pass)
				pass_oneline_num += 1
			elif len(funcstring) < DEL_TRACE_LINE_NUM and i > num_of_tests:   ###x行以下のトレースを削除する(fail)
				fail_oneline_num += 1
			else:
				func_sum.append(funcstring_copy)
				#func_name.append(funcstring)
				#func_return.append(returnarray)
				#func_argument.append(argarray)
		else:
			func_sum.append(funcstring_copy)
			#func_name.append(funcstring)
			#func_return.append(returnarray)
			#func_argument.append(argarray)

		if i % 100 == 0:
			print(i, "is complete\n")

	if W2VEC_LEARNING:
		print("Start Word2Vec Learning...")
		start = time.time()
		if W2VEC_FULL_ENCODING:
			model_name = word2vec.Word2Vec(sentences=func_sum, vector_size=W_V, window=WINDOW_, min_count=1, epochs=500, sg=1)  #epoch=270
		else:
			model_name = word2vec.Word2Vec(sentences=func_name, vector_size=W_V_NAME, window=WINDOW_, min_count=1, epochs=270, sg=1)  #epoch=270 ###関数名のみの符号化
		'''
		print("Function name is encoded.\n")
		model_return = word2vec.Word2Vec(sentences=func_return, vector_size=W_V_RETURN, window=WINDOW_, min_count=1, epochs=270, sg=1)  #epoch=270
		print("Return values is encoded.\n")
		model_argument = word2vec.Word2Vec(sentences=func_argument, vector_size=W_V_ARGUMENT, window=WINDOW_, min_count=1, epochs=270, sg=1)  #epoch=270
		print("Arguments is encoded.\n")
		'''
		word2vec_conv_time = time.time() - start
		if WRITE_TIME_INFORMATION:
			timewrite.write_time_w2v(word2vec_conv_time, TRACE_VERSION)
		print("convoltion time: ", word2vec_conv_time / 60)
		if USE_ZIKKEN_MODE:
			if os.path.exists(TRACE_VERSION + "/MODEL/" + "{}".format(seed)):
				print("ERROR: OVERWRITE RISK OCCURED")
				print(TRACE_VERSION + "/MODEL/" + "{}".format(seed) + "  is exists.")
				sys.exit()
			else:
				if not os.path.exists(TRACE_VERSION + "/MODEL/"):
					os.mkdir(TRACE_VERSION + "/MODEL/")
				os.mkdir(TRACE_VERSION + "/MODEL/" + "{}".format(seed))
				model_name.save(TRACE_VERSION + "/MODEL/" + "{}".format(seed) + W2V_MODEL_NAME_NAME)
				#model_return.save(TRACE_VERSION + "/MODEL/" + "{}".format(seed) + W2V_MODEL_NAME_RETURN)
				#model_argument.save(TRACE_VERSION + "/MODEL/" + "{}".format(seed) + W2V_MODEL_NAME_ARGUMENTS)
		else:
			#model.save(MODEL_SAVE_PATH)
			print("This mode cannot be used.\n")   ###未完成なので使えない。使わなくてもいいので未完成でもよい
			sys.exit()
		print("Save complete Word2Vec model.")
	else:
		MODEL_LOAD_PATH = TRACE_VERSION + "/MODEL/" + "{}".format(seed) + W2V_MODEL_NAME_NAME
		model_name = Word2Vec.load(MODEL_LOAD_PATH)
		'''
		MODEL_LOAD_PATH = TRACE_VERSION + "/MODEL/" + "{}".format(seed) + W2V_MODEL_NAME_RETURN
		model_return = Word2Vec.load(MODEL_LOAD_PATH)
		MODEL_LOAD_PATH = TRACE_VERSION + "/MODEL/" + "{}".format(seed) + W2V_MODEL_NAME_ARGUMENTS
		model_argument = Word2Vec.load(MODEL_LOAD_PATH)
		'''

	return func_sum, func_name, func_return, func_argument, model_name, pass_oneline_num, fail_oneline_num

def isfloat(s):  # 浮動小数点数値かどうか判定
    try:
        float(s)
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

def process_data(execution_trace_list, model_name, encoding_size):   ###すべてのトレースに対してテンソル型に変換する関数
	trace_tensor_list = []
	string_change_time = 0
	#print("lenfuncsum:",len(func_sum))
	#print(func_sum['label'])
	trace_index = 0
	for input_trace_data in execution_trace_list:
		trace_index += 1
		data = {"li": [], "label": 'pass', "data_index": trace_index}
		for vec_index, Li in enumerate(input_trace_data['data_name']):
			Li_vec_name = model_name.wv[Li]
			#Li_vec_return = model_return.wv[input_trace_data['data_return'][vec_index]]
			#Li_vec_argument = model_argument.wv[input_trace_data['data_argument'][vec_index]]
			start = time.time()
			Li_tensor_name = str_to_tensor(Li_vec_name, W_V)
			#Li_tensor_name = str_to_tensor(Li_vec_name, W_V_NAME)
			#Li_tensor_return = str_to_tensor(Li_vec_return, W_V_RETURN)
			#Li_tensor_argument = str_to_tensor(Li_vec_argument, W_V_ARGUMENT)
			string_change_time += time.time() - start
			#print("index:",index)
			#Li_tensor = torch.cat((Li_tensor_name, Li_tensor_return, Li_tensor_argument), 2)
			Li_tensor = Li_tensor_name
			data['li'].append(Li_tensor)
			#print(Li_tensor.shape)
		if input_trace_data['label'] == 'pass':
			data['label'] = 'pass'
		else:
			data['label'] = 'fail'

		trace_tensor_list.append(data)
		if trace_index % 10 == 0:
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
		tensor = torch.randn(int(len(float_list) / encoding_size), encoding_size)
		#print(tensor.shape)
		#print(len(float_list))
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
	if BiDIRECT_LSTM:   ###双方向LSTMの場合は用意する隠れ層のサイズが増える
		h0 = torch.randn(2, 1, 256)   ###こことmake_model関数がマジックナンバー化している (256の値はグローバル変数にしておくべきである。ただし、BidirectionalLSTMを使用する際のAttention Vectorのサイズはこの値の倍になる。ex. h0(2, 1, 256)→attention vector(512))
		c0 = torch.randn(2, 1, 256)
	else:
		h0 = torch.randn(1, 1, LSTM_OUTPUT_SIZE)
		c0 = torch.randn(1, 1, LSTM_OUTPUT_SIZE)
	h0 = try_gpu(h0)
	c0 = try_gpu(c0)
	randomtensor = torch.randn(len(func_sum['li']), 1, W_V)
	randomtensor = try_gpu(randomtensor)
	#randomtensor = randomtensor.cuda()
	for ind, d in enumerate(func_sum['li']):
		randomtensor[ind] = d
	'''   ###↑の処理内容
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

def compute_attention_forward(model, lstm_hidden_vector_list):   ###attention機構を用いてLSTMが出力する隠れ層への重み付けを行い、トレースを１つのベクトルにする関数
	#print(lstm_hidden_vector_list)
	#attention_vector = model['attention'].weight   ###attention機構における重みを計算するためのアテンションベクトル
	attention_vector = lstm_hidden_vector_list[-1]
	attention_vector = attention_vector.squeeze()   ###1次元削減([0]と同じ)
	attention_vector = try_gpu(attention_vector)

	attention_vector_dot_hidden_list = []
	for hs in lstm_hidden_vector_list[:-1]:
		hs = try_gpu(hs.squeeze())
		dot_compute_vector = torch.dot(hs, attention_vector)   ###LSTMのレイヤーにおける隠れ層とアテンションベクトルの内積
		#dot_compute_vector = cos_sim_torch(hs, attention_vector)   ###LSTMの各レイヤーにおける隠れ層とアテンションベクトルのコサイン類似度
		attention_vector_dot_hidden_list.append(dot_compute_vector)

	attention_vector_dot_hidden_tensor = torch.tensor(attention_vector_dot_hidden_list).unsqueeze(0)
	###unsqueezeは次元を操作する処理。shapeが変わる

	softmax_attention = nn.Softmax(dim=1)   ###内積した値の加重和を計算するためのsoftmax
	#print(attention_vector_dot_hidden_tensor.shape)
	attention_weight = softmax_attention(attention_vector_dot_hidden_tensor)   ###Attention VeectorとLSTM出力値との内積のSoftmaxがAttention重み
	#print(attention_weight)

	zero_list = [0] * LSTM_OUTPUT_SIZE   #LSTMの各レイヤーの隠れ層にアテンションウェイトをかけたものの総和が１つのトレースを表すベクトルになる([1, 256])
	#print(zero_list)
	return_input_vector = torch.tensor(zero_list)
	return_input_vector = try_gpu(return_input_vector)

	for hs_index, hs in enumerate(lstm_hidden_vector_list[:-1]):
		hs = try_gpu(hs)
		#print(hs.shape, attention_weight.shape)
		weight_list = [attention_weight[0][hs_index]] * LSTM_OUTPUT_SIZE   ###[attention weight]が行方向にLSTM hidden sizeの数�?け並ぶ(shape[LSTM_HIDDEN_V])
		weight_tensor = torch.tensor(weight_list)
		weight_tensor = try_gpu(weight_tensor)
		#print(hs.shape, weight_tensor.shape)   ###shape[1, 256], shape[256]
		#print((hs * weight_tensor).shape)   ###torch.Size([1, 256])
		return_input_vector = torch.add( return_input_vector, hs * weight_tensor )   ###LSTMの隠れ層とアテンションウェイトをかけたものを加算する
		#return_input_vector = torch.mul( return_input_vector, hs * weight_tensor )   ###LSTMの隠れ層とアテンションウェイトをかけたものを乗算する

	#print(return_input_vector.shape)

	return return_input_vector

def set_train_params(opt, learning_rate, loss, model, loss_weights):

	param_list = []
	param_list += model['line'].parameters()
	if ATTENTION_ENABLE:
		#param_list += model['attention'].parameters()
		pass
	#'''
	for l in model['mlp']:
		#print("model=\n",this.model)  #my change
		print("l:",l)
		if isinstance(l, list):
			for item in l:
				param_list += list(item.parameters())
				#print(item.weight)
		else:
			#print(l.weight)
			param_list += list(l.parameters())
		#print("param=\n",param_list)  #my change
	#for pr in param_list:   ###パラメータの閲覧
	#	print(pr)
	#'''

	if opt == "Adam" or opt == "ADAM" or opt == "adam":
		#optimizer = optim.Adam(param_list, learning_rate, eps=0.001, amsgrad=True)   #1e-06 is effective
		optimizer = optim.Adam(param_list, learning_rate)   #False AMSgrad and not tuning epsilon
	elif opt == "SGD" or opt == "sgd" or opt == "Sgd":
		optimizer = optim.SGD(param_list, learning_rate)
	else:
		assert False, ("Optimizer {} not supported".format(opt))

	if loss == "BCELoss" or loss == "bceloss" or loss == "BCELOSS" or loss == "BCEloss":
		#loss = {'function': nn.functional.binary_cross_entropy, 'weight': loss_weights}
		#if this.model[this.output_layer]['type'] == "sigmoid":
		print("Automatically combining output sigmoid and BCELoss to BCEWithLogitsLoss")
		loss = {'function': nn.functional.binary_cross_entropy_with_logits, 'weight': loss_weights}
		#prev_out_layer = this.output_layer
		#this.output_layer = this.model[this.output_layer]['input'][0]
		#del this.model[prev_out_layer]
	else:
		assert False, ("Loss function {} not supported".format(loss))

	return optimizer, loss

def execute_network(input_trace, model):

	input_trace, lstm_hidden_vector_list, string_change_time = lstm_network(input_trace, W_V, len(input_trace), model['line'])
	if ATTENTION_ENABLE:
		attention_input_vector = compute_attention_forward(model, lstm_hidden_vector_list)
		scores = mlp_forward(model, attention_input_vector)
	else:
		if MLP_INPUT_IS_AVERAGE:
			###LSTMの各隠れ層の値を平均化する処理
			mlp_input_vector = torch.tensor( [0] * LSTM_OUTPUT_SIZE )
			mlp_input_vector = try_gpu(mlp_input_vector)
			for hs_index, hs in enumerate(lstm_hidden_vector_list):
				mlp_input_vector = torch.add( mlp_input_vector, hs )
			mlp_input_vector = torch.div(mlp_input_vector, hs + 1)
			scores = mlp_forward(model, mlp_input_vector)
		else:
			scores = mlp_forward(model, input_trace['data'])
	#print(scores)
	return scores, input_trace, string_change_time
	#return input_trace, input_trace

def mlp_forward(model, inp_tensor):

	test_tensor(inp_tensor)

	out = model['mlp'][0](inp_tensor)
	#out = model['sigmoid'](out)   #不要な処理(BCEwith logits を参照)
	test_tensor(out)
	for i in range(1, len(model['mlp'])):
		out = model['mlp'][i](out)
		#out = model['sigmoid'](out)
		test_tensor(out)
	#out = model['sigmoid'](out)   #不要な処理(BCEwith logits を参照)

	return out

def output_set_labelling(datapoint_set, model, ep, name):

	p_list = []
	lstm_vector = []
	lstm_v_target_val = np.array([])
	lstm_v_target_fail = np.array([])   #tuikabun
	with torch.no_grad():
		pass_match, fail_match, pass_total, fail_total = 0, 0, 0, 0
		for d_index, datapoint in enumerate(datapoint_set):
			if d_index % 50 == 0:
				print(d_index, "is complete.")
			d_output, lstm_v, string_change_time = execute_network(datapoint, model)
			lstm_vector.append(lstm_v)
			if "pass" in datapoint['label']:
				lstm_v_target_val = np.append(lstm_v_target_val, 0)
				pass_total += 1
				p_list.append("{} 1".format(str(d_output.item())) )
				if d_output.item() >= 0.5:
					pass_match += 1
			elif "fail" in datapoint['label']:
				#print(d_index)
				lstm_v_target_fail = np.append(lstm_v_target_fail, 1)
				fail_total += 1
				p_list.append("{} 0".format(str(d_output.item())) )
				if d_output.item() <= 0.5:
					fail_match += 1
			else:
				assert False, "Unrecognized label"

	return pass_match, pass_total, fail_match, fail_total, p_list#, val, Y_val, val_np


def make_training_set(data_pass, data_fail, index_list_pass, index_list_fail):
	tr_set = []
	val_set = []
	for i in index_list_pass:
		tr_set.append(data_pass[i - 1])
		data_pass.pop(i - 1)
	for i in index_list_fail:
		tr_set.append(data_fail[i - 1])
		data_fail.pop(i - 1)
	val_set = data_pass + data_fail
	return tr_set, val_set

def make_trainingset(training_length, data_pass, data_fail):
	#トレーニングセットと評価用データ作成とデータ数の調整
	setting_num = int(training_length * (len(data_pass) + len(data_fail)))   #トレーニングデータの数
	training_set = data_pass[:int(training_length*len(data_pass))] + data_fail[:int(training_length*len(data_fail))]   #こっちが通常のサンプリング
	validation_set = data_pass[int(training_length*len(data_pass)):] + data_fail[int(training_length*len(data_fail)):]   #こっちが通常のサンプリング
	print("training pass", len(data_pass[:int(training_length*len(data_pass))]), "training fail:", len(data_fail[:int(training_length*len(data_fail))]), "val pass:", len(data_pass[int(training_length*len(data_pass)):]), "val fail:", len(data_fail[int(training_length*len(data_fail)):]))

	if len(training_set) < setting_num:   #訓練データの数が1足りない場合
		if int(training_length*len(data_pass)) < int(training_length*len(data_fail)):   #failが多い場合にpassを1増やす
			training_set = data_pass[:int(training_length*len(data_pass)) + 1] + data_fail[:int(training_length*len(data_fail))]
			validation_set = data_pass[int(training_length*len(data_pass)) + 1:] + data_fail[int(training_length*len(data_fail)):]
			print("training pass", len(data_pass[:int(training_length*len(data_pass)) + 1]), "training fail:", len(data_fail[:int(training_length*len(data_fail))]), "val pass:", len(data_pass[int(training_length*len(data_pass)) + 1:]), "val fail:", len(data_fail[int(training_length*len(data_fail)):]))

		else:   #passが多い場合にfailを1増やす
			training_set = data_pass[:int(training_length*len(data_pass))] + data_fail[:int(training_length*len(data_fail)) + 1]
			validation_set = data_pass[int(training_length*len(data_pass)):] + data_fail[int(training_length*len(data_fail)) + 1:]
			print("training pass", len(data_pass[:int(training_length*len(data_pass))]), "training fail:", len(data_fail[:int(training_length*len(data_fail)) + 1]), "val pass:", len(data_pass[int(training_length*len(data_pass)):]), "val fail:", len(data_fail[int(training_length*len(data_fail)) + 1:]))
	return training_set, validation_set


def train_dispatch(optimizer, loss_function, epochs, training_length, data_pass, data_fail, model_pass, model_fail, model, dataseed, print_to_out=True, ART=False, WOFUT=False, DATA_SEP_SET=False):

	###受け取っているdata_pass data_failは関数名をWord2Vecで符号化したものなので注意するように。基本的にこれ以降の処理でWrod2Vecモデルは使用しない。
	#'''   ###データシャッフル、ランダムサンプリング, 交差検証で使用, ART,WOFUTでは使わない
	random.seed(100)
	#data_plus = data_fail + data_pass
	for i in range(dataseed):
		shuffle(data_pass)
		shuffle(data_fail)
		#shuffle(data_plus)
	#'''
	oppoint = []
	summax = 0
	summaxstr = ""
	print(model)
	if DATA_SEP_SET:
		training_set, validation_set = make_sep_trainingset(data_pass, data_fail, PASS_SEP, FAIL_SEP)
	elif TRAINING_FULL_DATA:   ###全てのデータが訓練データとなり、評価用データも同じデータ構造
		training_set = data_pass + data_fail
		validation_set = training_set
	else:
		training_set, validation_set = make_trainingset(training_length, data_pass, data_fail)

	if USE_ZIKKEN_MODE:
		if os.path.exists(TRACE_VERSION + "/MODEL/" + "{}".format(dataseed)):
			train_log_path = TRACE_VERSION + "/MODEL/" + str(dataseed) + "/" + TRAIN_LOG_NAME
			#print("ERROR: Designated dir exists !!!\n It is possible to Overwite.")
			#return
		else:
			print("ERROR: dir is not exists")
			print(TRACE_VERSION + "/MODEL/" + "{}".format(dataseed))
			sys.exits()
		#train_log_path = TRACE_VERSION + "/MODEL/" + str(dataseed) + "/" + TRAIN_LOG_NAME
		tf = open(train_log_path, "a", encoding="utf-8")
	else:
		tf = open(TRAIN_LOG_PATH, "a", encoding="utf-8")

	if USE_ZIKKEN_MODE:
		tf.write(str(model) + "\n" + TRACE_VERSION + "\n" + "training traces path:\n" + PASS_TRACES + "\n" + FAIL_TRACES + "\n" + "ATTENTION_ENABLE: " + str(ATTENTION_ENABLE) + "\nMLP_INPUT_IS_AVERAGE: " + str(MLP_INPUT_IS_AVERAGE) + "\nWord2Vec MODEL:" + TRACE_VERSION + "/MODEL/" + "{}".format(dataseed) + W2V_MODEL_NAME_NAME + "\nWord2Vec Window size: " + str(WINDOW_) + "\nLearning Rate: " + str(LEARNING_RATE_) + "\nTraining size: " + str(TRAINING_SIZE_) + "\nTrainingu Full Data: " + str(TRAINING_FULL_DATA) + "\nSAVE POINT: " + str(SAVE_POINT) + "\n")
	else:
		print("USE_ZIKKEN_MODE False is cannot use.\n")   ###グローバル変数MODEL_LOAD_PATHの内容が以前のままのため、三変数分割に対応していない。したがって使用不可
		sys.exit()

	learning_time = 0
	inference_time_test = 0
	inference_time_val = 0
	total_string_list_change_time = 0

	model_save_count = 0   ###保存したPymodelの数を格納する変数

	for ep in range(epochs):

		print("Epoch {}".format(ep))
		epoch_stats = "Epoch {}\n----------\n".format(ep)

		start = time.time()
		################################クラスタリングする際に、どのトレースが訓練でータになるかを観察する際はここのシャフルはコメントアウトしたほうが良い?はず。（クラスタリングした段階でprintしてるから大丈夫。要確認?
		for i in range(100):
			shuffle(training_set)
		#lstm_vector = []
		for tr_index, tr_data in enumerate(training_set):
			#print(tr_data)
			out, lstm_v, string_change_time = execute_network(tr_data, model)
			total_string_list_change_time += string_change_time

			if "pass" in tr_data['label']:
				target = torch.tensor([[1.]])
				target = try_gpu(target)
				weight = loss_function['weight']['pass']
				weight = try_gpu(weight)
			elif "fail" in tr_data['label']:
				target = torch.tensor([[0.]])
				target = try_gpu(target)
				weight = loss_function['weight']['fail']
				weight = try_gpu(weight)
			else:
				assert False, "Unrecognized label"

			loss = loss_function['function'](out, target, weight=weight)
			#loss.backward(retain_graph=True)
			loss.backward()
			optimizer.step()

			### Report Loss every now and then
			if (tr_index % 10) == 0 and print_to_out == True:
				print("{} loss: {:6.4f}, {}".format(tr_data['label'], loss.item(), tr_index))
			elif tr_data['label'] == 'fail':
				print("{} loss: {:6.4f}, {}".format(tr_data['label'], loss.item(), tr_index))
			elif tr_data['label'] == 'pass':
				print("{} loss: {:6.4f}, {}".format(tr_data['label'], loss.item(), tr_index))

		learning_time += time.time() - start

		#### Training Set inference
		start = time.time()   #trainingを評価する場合コメントアウトを削除
		pass_match, pass_total, fail_match, fail_total, _, = output_set_labelling(training_set, model, ep, "train")
		inference_time_test += time.time() - start

		print("\nTraining set:\npass matches: {}\npass total: {}\nfail matches: {}\nfail total: {}\n".format(pass_match, pass_total, fail_match, fail_total))
		tf.write("\nepoch:" + str(ep) +  "\nTraining set:\npass matches: " +  str(pass_match) +  "\npass total: " +  str(pass_total) + "\nfail matches: " +  str(fail_match) +  "\nfail total: " +  str(fail_total) + "\n")


		### Validation set inference
		#'''   ##validationを評価する場合はコメントアウトを削除
		start = time.time()
		pass_match, pass_total, fail_match, fail_total, _ = output_set_labelling(validation_set, model, ep, "vali")
		inference_time_val += time.time() - start

		print("\nValidation set:\npass matches: {}\npass total: {}\nfail matches: {}\nfail total: {}\n".format(pass_match, pass_total, fail_match, fail_total))

		tf.write("\nValidation set:\npass matches: " +  str(pass_match) +  "\npass total: " +  str(pass_total) + "\nfail matches: " +  str(fail_match) +  "\nfail total: " +  str(fail_total) + "\n")
		#'''   #validationを評価する場合コメントアウトを削除
		if TRAINING_FULL_DATA:   ###全てのデータが訓練データの場合、評価用データの分類時間は無駄
			pass_match, pass_total, fail_match, fail_total, _ = pass_match, pass_total, fail_match, fail_total, _
		if fail_total != 0:
			recall_a = fail_match / fail_total
		else:
			recall_a = 0
		if pass_total != 0:
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
		tf.write("Precision: " +  str(precision_a) + "\nRecall: " +  str(recall_a) +  "\nTNR: " + str(TNR_a) +  "\nF-measure:" +  str(f_measure) + "\n")
		if precision_a > 0.85 and recall_a > 0.85:
			if TNR_a > 0.85:
				appestr = "\nepoch: "
				appestr += str(ep)
				appestr += "\npass: "
				appestr += str(pass_match)
				appestr += "\nfail: "
				appestr += str(fail_match)
				appestr += "\nPrecision: "
				appestr += str(precision_a)
				appestr += "\nRecall: "
				appestr += str(recall_a)
				appestr += "\nTNR: "
				appestr += str(TNR_a)
				appestr += "\n"
				oppoint.append(appestr)
		#sumpr = precision_a + recall_a
		sumpr = recall_a
		sumpr += TNR_a
		if summax <= sumpr:
			summaxstr = "epoch: " + str(ep) + "\npass: " + str(pass_match) + "\nfail " + str(fail_match) + "\nPrecision: " + str(precision_a) + "\nRecall: " + str(recall_a) + "\nTNR: " + str(TNR_a) + "\n"
			summax = sumpr

		save_point = SAVE_POINT
		'''
		if precision_a > save_point and recall_a > save_point and TNR_a > save_point and SAVE_PYMODEL_FLAG:
			if USE_ZIKKEN_MODE:
				model_save_path = TRACE_VERSION + "/MODEL/" + str(dataseed)
			else:
				model_save_path = PYMODEL_PATH
			os.mkdir(model_save_path + "/model_ep{}".format(ep))
			save_pymodel(model_save_path + "/model_ep{}/".format(ep), model)
			print("complete save pymodel")
			#break
		'''

		if recall_a > (save_point + 0) and TNR_a > (save_point + 0) and SAVE_PYMODEL_FLAG:
			if USE_ZIKKEN_MODE:
				model_save_path = TRACE_VERSION + "/MODEL/" + str(dataseed)
			else:
				model_save_path = PYMODEL_PATH
			if model_save_count <= 3:
				os.mkdir(model_save_path + "/model_ep{}".format(ep))
				save_pymodel(model_save_path + "/model_ep{}/".format(ep), model)
				print("complete save pymodel")
				model_save_count += 1
			#break
		if model_save_count >= 4:
			break
		#pca_exe(lstm_vector)
	tf.write("\nOPT POINT:\n")
	print(oppoint)
	for opt in oppoint:
		tf.write(opt)
	tf.write("MAX POINT:\n" + summaxstr)
	tf.close()

	print("learning time:",learning_time / 60)
	if WRITE_TIME_INFORMATION:   ###学習時間のログファイルへの書き込み
		timewrite.write_time_imformation(learning_time, TRACE_VERSION)
	print("total_string_list_change_time:", total_string_list_change_time / 60)
	print("string change time is excluded time:", (learning_time - total_string_list_change_time) / 60)
	print("inference test time:",inference_time_test / 60)
	print("inference val time:",inference_time_val / 60)
	#op.close()

def save_pymodel(pymodel_path, model):

	print("Saving the model....")
	torch.save(model['line'].state_dict(), pymodel_path + "line" + ".pth")
	if ATTENTION_ENABLE:
		#torch.save(model['attention'].state_dict(), pymodel_path + "attention" + ".pth")
		pass

	for index, hidden in enumerate(model['mlp']):
		#print(hidden.state_dict())
		torch.save(hidden.state_dict(), pymodel_path + 'mlp' + "_" + str(index) + ".pth")

	return 0

def make_model(input_len):
	out_feature = [256, 128, 64, 1]   ###256, 128, 64, 1
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
								hidden_size = input_len,
								num_layers = 1,
								bias = False,
								batch_first = False,
								dropout = 0,
								bidirectional = BiDIRECT_LSTM)
	else:
		if BiDIRECT_LSTM:
			nn_layer = nn.LSTM(input_size = W_V,
								hidden_size = 256,   ###他機能と整合性を取るためここだけ条件分岐、改善可能 (LSTM_OUTPUT_SIZE / 2)に設定する必要がある
								num_layers = 1,
								bias = False,
								batch_first = False,
								dropout = 0,
								bidirectional = BiDIRECT_LSTM)
		else:
			nn_layer = nn.LSTM(input_size = W_V,
								hidden_size = input_len,
								num_layers = 1,
								bias = False,
								batch_first = False,
								dropout = 0,
								bidirectional = BiDIRECT_LSTM)

	nn_layer = try_gpu(nn_layer)
	#nn_layer.eval()

	#sigmoid = nn.Sigmoid()
	#model = {'line': nn_layer, 'mlp': [], 'sigmoid': sigmoid}
	if ATTENTION_ENABLE:  ##不要、Attention用にレイヤー作成するのはそこまでよくなかった
		#attention_layer = nn.Linear(in_features = input_len,
		#							out_features = 1,
		#							bias = False)
		#model = {'line': nn_layer, 'mlp': [], "attention": attention_layer}
		model = {'line': nn_layer, 'mlp': []}
	else:
		model = {'line': nn_layer, 'mlp': []}

	linear_layer = nn.Linear(in_features = input_len,
									out_features = out_feature[0],
									bias = False)
	linear_layer = try_gpu(linear_layer)
	#linear_layer.eval()
	model['mlp'].append(linear_layer)
	for i in range(1, len(out_feature)):
		linear_layer = nn.Linear(in_features = out_feature[i - 1],
										out_features = out_feature[i],
										bias = False)
		linear_layer = try_gpu(linear_layer)
		#linear_layer.eval()
		model['mlp'].append(linear_layer)
	return model

def weights_cal(pass_len, fail_len):
	loss_weights = {"pass": 0.0, "fail": 0.0}
	loss_weights['fail'] = torch.FloatTensor([pass_len / (pass_len + fail_len)])
	loss_weights['pass'] = torch.FloatTensor([fail_len / (pass_len + fail_len)])
	#loss_weights['fail'] = torch.FloatTensor([0.5])
	#loss_weights['pass'] = torch.FloatTensor([0.5])
	return loss_weights

def edit_func_sum(data_name, label):
	edit_data = []
	for index, seq in enumerate(data_name):
		edit_data.append({'data_name': [], 'label': 'fail', 'data_index': index + 1})
		for li in seq:
			edit_data[index]['data_name'].append(li)
		'''
		for li in data_return[index]:
			edit_data[index]['data_return'].append(li)
		for li in data_argument[index]:
			edit_data[index]['data_argument'].append(li)
		'''
		if label == "pass":
			edit_data[index]["label"] = "pass"
		else:
			edit_data[index]["label"] = "fail"

	return edit_data

def main(s):

	passnum = PASS_TR_NUM
	failnum = FAIL_TR_NUM
	training_size = TRAINING_SIZE_
	func_sum_total, func_name, func_return, func_argument, model_name, pass_oneline_num_r, fail_oneline_num_r = pool_trace(PASS_TRACES, REDUCED_PASS_TRACES, passnum, FAIL_TRACES, REDUCED_FAIL_TRACES, failnum, returnencode=False, argencode=False, callerencode=False, calledencode=False, returnrate=1, argrate=1, seed=s, trlength=training_size)

	if pass_oneline_num_r > 0:
		func_sum_pass_total = func_sum_total[:(passnum - pass_oneline_num_r)]
		func_sum_fail_total = func_sum_total[(passnum - pass_oneline_num_r):]
		'''
		func_sum_pass_name = func_name[:(passnum - pass_oneline_num_r)]
		func_sum_pass_return = func_return[:(passnum - pass_oneline_num_r)]
		func_sum_pass_argument = func_argument[:(passnum - pass_oneline_num_r)]
		func_sum_fail_name = func_name[(passnum - pass_oneline_num_r):]
		func_sum_fail_return = func_return[(passnum - pass_oneline_num_r):]
		func_sum_fail_argument = func_argument[(passnum - pass_oneline_num_r):]
		'''
	else:
		func_sum_pass_total = func_sum_total[:passnum]
		func_sum_fail_total = func_sum_total[passnum:]
		'''
		func_sum_pass_name = func_name[:passnum]
		func_sum_pass_return = func_return[:passnum]
		func_sum_pass_argument = func_argument[:passnum]
		func_sum_fail_name = func_name[passnum:]
		func_sum_fail_return = func_return[passnum:]
		func_sum_fail_argument = func_argument[passnum:]
		'''

	#'''
	func_sum_pass = edit_func_sum(func_sum_pass_total, "pass")   ###三つの変数がディレクトリ方式で格納されているリストが返される
	func_sum_fail = edit_func_sum(func_sum_fail_total, "fail")
	network_model = make_model(LSTM_OUTPUT_SIZE)   ###引数はLSTMの出力サイズ ###256
	loss_weights = weights_cal(len(func_sum_pass), len(func_sum_fail))
	opt, loss = set_train_params("Adam", LEARNING_RATE_, "BCELoss", network_model, loss_weights)   #lr=0.0001

	dataseed = s
	print("Preprocess Execution Trace Data...")
	func_sum_pass_processed = process_data(func_sum_pass, model_name, W_V_NAME + W_V_RETURN + W_V_ARGUMENT)
	func_sum_fail_processed = process_data(func_sum_fail, model_name, W_V_NAME + W_V_RETURN + W_V_ARGUMENT)
	train_dispatch(opt, loss, EPOCH_NUMBER, training_size, func_sum_pass_processed, func_sum_fail_processed, model_name, model_name, network_model, dataseed, ART=False, WOFUT=False, DATA_SEP_SET=False)   ###実行トレースの前処理を事前に行う処理の追加によってクラスタリングとARTは使用不可。要アプデ
	return

#'''
if __name__ == "__main__":   #monte carlo cross validtion
	seedlist = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
	#seedlist = [12]
	#seedlist = [22, 182]
	seedlist = [i for i in range(500, 560, 20)]
	#seedlist = [500]
	#seedlist = [200, 201, 202, 203, 204, 205, 206, 207, 208, 209]
	#seedlist = [i for i in range(211, 311, 20)]
	seedlist = [SEED_VALUE]
	# seedlist = [i for i in range(30, 35, 1)]

	for s in seedlist:
		elpased_start = time.time()
		main(s)
		elapsed_time = time.time() - elpased_start
		min = elapsed_time / 60
		sec = elapsed_time % 60
		print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
		print ("elapsed_time:",min,"[min]",sec,"[sec]")
