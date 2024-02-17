#!/usr/bin/env python
import time
import sys
import subprocess
from subprocess import PIPE
import os
from os import environ as get_env
import shutil
from concurrent import futures
import argparse
from threading import Lock

from lxml import html
import numpy as np
import glob
from math import log10

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)


class Color:
    BLACK = '\033[30m'  # (文字)黒
    RED = '\033[31m'  # (文字)赤
    GREEN = '\033[32m'  # (文字)緑
    YELLOW = '\033[33m'  # (文字)黄
    BLUE = '\033[34m'  # (文字)青
    MAGENTA = '\033[35m'  # (文字)マゼンタ
    CYAN = '\033[36m'  # (文字)シアン
    WHITE = '\033[37m'  # (文字)白
    REVERSE = '\033[07m'  # 文字色と背景色を反転
    DEFAULT = '\033[39m'  # 文字色をデフォルトに戻す
    RESET = '\033[0m'  # 全てリセット


class Printer_With_Locker():
    def __init__(self, max_tasks):
        self.executed = 0
        self.max_tasks = max_tasks
        self.succeed = "\rDone - {:" + \
            str(int(log10(max_tasks))+1)+"}/"+str(max_tasks)
        self.lock = Lock()

    def reset(self):
        self.executed = 0

    def successed(self):
        with self.lock:
            self.executed += 1
            filler = int(self.executed/self.max_tasks * 100 / 2.5)
            print(self.succeed.format(self.executed) +
                  "[" + "="*filler + " "*(int(100/2.5)-filler) + "]", end="", flush=True)


parser = argparse.ArgumentParser(description='collect each test coverage.')
parser.add_argument('project_id', help='project ID of defects4j')
parser.add_argument('bug_id', help='bug ID of defects4j')
# parser.add_argument('target', help='directory name of build dir')
# parser.add_argument('classes', help='Directory name of classes')
# parser.add_argument('tests', help='directory name of tests')
# parser.add_argument('source_dir', help='source directory of the project')
parser.add_argument('-c', '--only_get_coverage',
                    action='store_true', help='collect only coverage')
args = parser.parse_args()
# args
# BUILD_dir
##
script_dir = "/tmp/" + args.project_id + "/" + args.bug_id + "/"
query_path = "/tmp/" + args.project_id + "/query.txt"
temp_exec_dir = script_dir + "exec/"
report_dir = script_dir + "report/"
cov_output_dir = script_dir.replace("tmp/", "workspace/coverage/")
single_test_runner_path = "/workspace/bin"
agent_jar = "/root/jacoco/lib/jacocoagent.jar"
cli_jar = "/root/jacoco/lib/jacococli.jar"

# get variables depending on projects
os.chdir(script_dir)
build_dir = subprocess.run(["defects4j export -p dir.bin.classes"],
                           capture_output=True, text=True, shell=True).stdout.split("/")
test_cp = subprocess.run(["defects4j export -p cp.test"],
                         capture_output=True, text=True, shell=True).stdout.replace(build_dir[0], "instrumented")
# build_dir.insert(0, "target")
target_dir = script_dir + build_dir[0] + "/" + build_dir[1] + "/"
class_dir = script_dir + "instrumented/" + build_dir[0] + "/"
# test_dir = script_dir + "instrumented/" + \
#     subprocess.run(["defects4j export -p dir.bin.tests"], capture_output=True,
                #    text=True, shell=True).stdout.split("/")[1] + "/"
# chartのみ
# test_dir = script_dir + "instrumented/" + subprocess.run(["defects4j export -p dir.bin.tests"], capture_output=True, text=True, shell=True).stdout

source_dir = script_dir + subprocess.run(["defects4j export -p dir.src.classes"], capture_output=True, text=True, shell=True).stdout + "/"

# collct JUnit tests
test_file_path = script_dir + "all_tests.txt"
test_list = []
with open(test_file_path) as file:
    test_list = list(map(lambda s: s.replace(
        "\n", "").split(",")[1], file.readlines()))
num_all_test = len(test_list)

printer = Printer_With_Locker(num_all_test)
# set path to taget class and source code

test_failures = []
errs = []

target_class = ""
target_source = ""
with open(query_path) as file:
    for s in file.readlines():
        if s.startswith(str(args.bug_id)+","):
            base_name = s.replace("\n", "").split(
                ",")[1].replace(".", "/").replace("\"", "")
            # 複数バグの場合, ;で分割して各クラスの場所リストを作る
            #base_name = base_name.split(";")
            #arget_class = [target_dir + base_n + ".class" for base_n in base_name]
            target_class = target_dir + base_name + ".class"
            target_source = source_dir + base_name + ".java"
            break
target_base_name = target_source.split("/")[-1]
LOC = 0
with open(target_source, mode="r") as file:
    LOC = len(file.readlines())


def run_test(test, index):
    exec_file = temp_exec_dir + test + ".exec"

    # run test
    command_run_junit_test = "java -Djacoco-agent.destfile=\"{}\" -cp {}:{}:{} SingleJUnitTestRunner {}".format(
        exec_file, test_cp, single_test_runner_path, agent_jar, test)
    proc = subprocess.run([command_run_junit_test],
                          capture_output=True, text=True, shell=True)
    if not proc.stdout.endswith("successed\n"):
        test_failures.append(
            (command_run_junit_test, proc.stdout, proc.stderr))

    # extract a report that relates to the target class from exec file
    report_path = report_dir + \
        ("pass" if proc.stdout.endswith("successed\n")
         else "fail") + "_" + test + ".xml"
    command_extract_report = "java -jar {} report {} --classfiles {} --xml {} --sourcefiles {}".format(
        cli_jar, exec_file, target_class, report_path, source_dir)
    proc = subprocess.run([command_extract_report],
                          capture_output=True, text=True, shell=True)

    if proc.returncode != 0:
        errs.append((command_extract_report, proc.stdout, proc.stderr))

    global printer
    printer.successed()


# make empty the exec directory
if os.path.exists(temp_exec_dir):
    shutil.rmtree(temp_exec_dir)
os.makedirs(temp_exec_dir)
if not args.only_get_coverage:
    # make empty the report directory
    if os.path.exists(report_dir):
        shutil.rmtree(report_dir)
    os.makedirs(report_dir)

    # make empty the output directory
    if os.path.exists(cov_output_dir):
        shutil.rmtree(cov_output_dir)
    os.makedirs(cov_output_dir)

    failed_tests = ""

    job_list = []
    try:
        with futures.ThreadPoolExecutor() as executor:
            TEST_RUN = True
            if TEST_RUN:
                job_list = [executor.submit(run_test, index=index, test=test)
                            for index, test in enumerate(test_list)]
            else:
                num_all_test = 1000
                for index, test in enumerate(test_list):
                  if index >= num_all_test:
                    break
                  job_list.append(executor.submit(run_test, index=index, test=test))

            for fut in futures.as_completed(fs=job_list):
                fut.result()
    except KeyboardInterrupt:
        sys.exit(1)
    if failed_tests != "":
        with open(script_dir + "failed_tests.txt", mode="w") as f:
            f.write(failed_tests)

        # write logs
    print("\nfailed tests: " + str(len(test_failures)) +
            ", errors: " + str(len(errs)))

with open(script_dir+"failed_tests.txt", mode="w") as f:
    for failure in test_failures:
        f.write("\t[test name]\n" + failure[0] + "\n")
        f.write("\t[stdout]\n" + failure[1])
        f.write("\t[stderr]\n" + failure[2])
        f.write("=================\n")
if len(errs) != 0:
    with open(script_dir+"err_tests.txt", mode="w") as f:
        for e in errs:
            f.write("\t[test name]\n" + e[0] + "\n")
            f.write("\t[stdout]\n" + e[1])
            f.write("\t[stderr]\n" + e[2])
            f.write("=================\n")

# sys.exit(1)

trigger_tests = subprocess.run(["defects4j export -p tests.trigger"], capture_output=True,
                               text=True, shell=True).stdout.replace("::", "#").split("\n")

time.sleep(1)
# making is_plane.txt
with open(glob.glob(report_dir+"*.xml")[0], mode="r") as f:
    xml_str = f.read().replace("\n", "").replace("encoding=\"UTF-8\" ", "")
root = html.fromstring(xml_str)
executed_information = html.fromstring(xml_str).xpath(
    '//sourcefile[@name="' + target_source.split("/")[-1] + '"]')[0]
plane = np.ones(LOC)
for e in executed_information.xpath('./line'):
    plane[int(e.attrib['nr'])-1] = 0
np.savetxt(cov_output_dir + "is_plane.txt", plane, fmt="%1d")

pass_fail_counter = {"pass": 0, "fail": 0}
printer.reset()
print("Collecting Coverages")
for index, report in enumerate(glob.glob(report_dir+"*.xml")):
    printer.successed()
    result = os.path.basename(report)[:4]
    if result == "fail" and os.path.basename(report)[5:-4] not in trigger_tests:
        continue

    with open(report, mode="r") as f:
        xml_str = f.read().replace("\n", "").replace("encoding=\"UTF-8\" ", "")
    root = html.fromstring(xml_str)

    executed_information = html.fromstring(xml_str).xpath(
        '//sourcefile[@name="' + target_source.split("/")[-1] + '"]')[0]
    executed_lines = int(executed_information.xpath(
        './counter[@type="LINE"]')[0].attrib['covered'])
    if executed_lines == 0:
        continue
    whole_lines = len(executed_information.xpath('./line'))

    coverage = np.zeros(whole_lines)
    for index, e in enumerate(executed_information.xpath('./line')):
        if e.attrib['ci'] != '0':
            coverage[index] = 1

    digit = int(log10(num_all_test))+1
    save_name = cov_output_dir + result + '_' + \
        ("{:0"+str(digit)+"d}").format(pass_fail_counter[result]) + '.csv'
    pass_fail_counter[result] += 1
    np.savetxt(save_name, coverage, fmt="%1d")
print()
print(pass_fail_counter)

result_dir = "/workspace/tmp/" + args.project_id + "/" + args.bug_id + "/"
if os.path.exists(result_dir):
    shutil.rmtree(result_dir)
os.makedirs(result_dir)
shutil.copytree(temp_exec_dir, result_dir+"exec")
shutil.copytree(report_dir, result_dir + "report")
