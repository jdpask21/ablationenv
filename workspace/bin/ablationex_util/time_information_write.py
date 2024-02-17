WRITE_FILE_PATH_TIME_INFORMATION = "time_information0123-pt-tr.log"

def write_time_imformation(inf_time, sut_version):   ###実行時間を書き込む関数
	wr_file_time = open(WRITE_FILE_PATH_TIME_INFORMATION, "a")
	wr_file_time.write(sut_version + "\n")
	wr_file_time.write(str(inf_time) + "\n")
	wr_file_time.close()
	return 0

def write_time_w2v(learning_time, sut_version):
	wr_file_time = open(WRITE_FILE_PATH_TIME_INFORMATION, "a")
	wr_file_time.write(sut_version + " Word2Vec\n")
	wr_file_time.write(str(learning_time) + "\n")
	wr_file_time.close()
	return 0
