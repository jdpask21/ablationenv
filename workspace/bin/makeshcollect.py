file_path = "collecttotinfo.txt"

file = open(file_path, "r")
#shfile = open("exe_checkoutsh.sh", "w")
shfile = open("exe_online_sir.sh", "a")
#shfile = open("exe_learning.sh", "w")
#shfile = open("exe_calochiai.sh", "a")
#shfile = open("exe_cehck_unique.sh", "a")
#shfile = open("exe_check_multi.sh", "a")

#exe_command = "python hoge.py -c "
#exe_command = "./checkout.sh "
#exe_command = "python main_line.py "
exe_command = "python onlin_line.py "
#exe_command = "python cal_ochiai.py "
#exe_command = "python check_multirepair.py "
target_start = "totinfo 2\n"

strat_flag = False

while (True):
    line = file.readline()
    if line:
        if line == target_start:
            strat_flag = True
        if strat_flag:
            pidandbid_list = line.split("\n")[0].split(" ")
            execute_write = exe_command + pidandbid_list[0] + " " +\
                  pidandbid_list[1] + "\n"
            shfile.write(execute_write)
    else:
        break
file.close()
shfile.close()