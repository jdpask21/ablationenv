#!/usr/bin/env python
import sys
import subprocess
import os
import shutil
import argparse
from concurrent import futures
from threading import Lock

from lxml import etree
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
        self.counter = {"pass": 0, "fail": 0}

    def successed(self):
        with self.lock:
            self.executed += 1
            filler = int(self.executed/self.max_tasks * 100 / 2.5)
            print(self.succeed.format(self.executed) +
                    "[" + "="*filler + " "*(int(100/2.5)-filler) + "]", end="", flush=True)

    def get_count(self, result, increment=False):
        with self.lock:
            ret = self.counter[result]
            if increment:
                self.counter[result] += 1
            return self.counter[result]

    def reset(self):
        self.executed = 0
        self.counter = {"pass": 0, "fail": 0}


parser = argparse.ArgumentParser(description='collect each test coverage.')
parser.add_argument('project_id', help='project ID of defects4j')
parser.add_argument('bug_id', help='bug ID of defects4j')
parser.add_argument('-c', '--only_get_coverage',
                    action='store_true', help='collect only coverage')
args = parser.parse_args()
# args
# BUILD_dir
##
script_dir = "/tmp/" + args.project_id + "/" + args.bug_id + "/"

report_dir = script_dir + "report/"
cov_output_dir = script_dir.replace("/tmp", "/workspace/clover")
single_test_runner_path = "/workspace/bin"
clover_jar = "/root/clover/lib/clover.jar"

# get variables depending on projects
os.chdir(script_dir)

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

src_dir = subprocess.run(["defects4j export -p dir.src.classes"], capture_output=True, text=True, shell=True).stdout + "/"

#modified_src = script_dir + "tmp_" + src_dir + \
#    subprocess.run(["defects4j export -p classes.modified"], capture_output=True, text=True, shell=True).stdout.replace(".", "/") + ".java"
###Math
#modified_src = script_dir + src_dir.replace("java", "tmp_java") + \
#    subprocess.run(["defects4j export -p classes.modified"], capture_output=True, text=True, shell=True).stdout.replace(".", "/") + ".java"

###Chart
#modified_src = script_dir + src_dir.replace("source", "tmp_source") + \
#    subprocess.run(["defects4j export -p classes.modified"], capture_output=True, text=True, shell=True).stdout.replace(".", "/") + ".java"
###Closure
modified_src = script_dir + src_dir.replace("source", "tmp_src") + \
    subprocess.run(["defects4j export -p classes.modified"], capture_output=True, text=True, shell=True).stdout.replace(".", "/") + ".java"

LOC = 0
with open(modified_src, mode="r") as file:
    LOC = len(file.readlines())

test_cp = subprocess.run(["defects4j export -p cp.test"], capture_output=True, text=True, shell=True).stdout
def run_test(test):
    # run test
    command_run_junit_test = "java -cp {}:{} SingleJUnitTestRunner {}".format(
        test_cp, single_test_runner_path, test)
    proc = subprocess.run([command_run_junit_test],
                          capture_output=True, text=True, shell=True)
    if not proc.stdout.endswith("successed\n"):
        test_failures.append(
            (command_run_junit_test, proc.stdout, proc.stderr))

    # extract a report that relates to the target class from xml file
    report_path = report_dir + ("pass" if proc.stdout.endswith("successed\n") else "fail") + "_" + test + ".xml"
    command_extract_report = "java -cp {} com.atlassian.clover.reporters.xml.XMLReporter -if -l -i clover.db -o {}".format(clover_jar, report_path)
    proc = subprocess.run([command_extract_report],
                          capture_output=True, text=True, shell=True)
    for file in glob.glob(script_dir+"clover.db*"):
        os.remove(file)
    shutil.copy("tmp_clover.db", "clover.db")

    global printer
    printer.successed()


if not args.only_get_coverage:
    #'''
    # make empty the report directory
    if os.path.exists(report_dir):
        shutil.rmtree(report_dir)
    os.makedirs(report_dir)
    #'''

    # make empty the output directory
    if os.path.exists(cov_output_dir):
        shutil.rmtree(cov_output_dir)
    os.makedirs(cov_output_dir)

    failed_tests = ""

    print("Collecting Coverages of {}-{}...".format(args.project_id, args.bug_id))
    #'''
    try:
        for test in test_list:
            run_test(test)
    except KeyboardInterrupt:
        sys.exit(1)
    if failed_tests != "":
        with open(script_dir + "failed_tests.txt", mode="w") as f:
            f.write(failed_tests)
    #'''
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

trigger_tests = subprocess.run(["defects4j export -p tests.trigger"], capture_output=True, text=True, shell=True).stdout.replace("::", "#").split("\n")


# making is_plane.txt
plane = np.ones(LOC)
root = etree.parse(glob.glob(report_dir+"*.xml")[0])
print(len(glob.glob(report_dir+"*.xml")))
for e in root.xpath('//file[@path="{}"]/line'.format(modified_src)):
    plane[int(e.attrib.get("num"))-1] = 0
np.savetxt(cov_output_dir + "is_plane.txt", plane, fmt="%1d")

pass_fail_counter = {"pass": 0, "fail": 0}
printer.reset()
print("Extracting Coverages of {}-{}...".format(args.project_id, args.bug_id))

def extract_coverage(report):
    result = os.path.basename(report)[:4]
    if result == "fail" and os.path.basename(report)[5:-4] not in trigger_tests:
        os.remove(report)
        return

    root = etree.parse(report).getroot()
    executed_file = root.xpath('//file[@path="{}"]'.format(modified_src))[0]
    executed_lines = executed_file.xpath('./line')
    with open(report, mode="wb") as f:
        etree.ElementTree(executed_file).write(f, encoding='utf-8', xml_declaration=True)

    coverage = np.zeros(len(set([e.attrib.get("num") for e in executed_lines])))
    cooverd = dict()
    index = -1
    for e in executed_lines:
        if not e.attrib.get("num") in cooverd:
            cooverd[e.attrib.get("num")] = True
            index += 1
        if e.attrib.get("type") == "cond":
            if e.attrib.get("truecount") != "0" or e.attrib.get("falsecount") != "0":
                coverage[index] = 1
        else:
            if e.attrib.get("count") != "0":
                coverage[index] = 1
    if not np.all(coverage == 0):
        printer.successed()
        digit = int(log10(num_all_test))+1
        save_name = cov_output_dir + result + '_' + \
            ("{:0"+str(digit)+"d}").format(printer.get_count(result, increment=True)) + '.csv'
        np.savetxt(save_name, coverage, fmt="%1d")

try:
    with futures.ThreadPoolExecutor() as executor:
        job_list = [executor.submit(extract_coverage, report=repo)
                    for repo in  glob.glob(report_dir+"*.xml")]
        for fut in futures.as_completed(fs=job_list):
            fut.result()
except KeyboardInterrupt:
    sys.exit(1)
print()
print(printer.counter)

result_dir = "/workspace/clover/tmp/" + args.project_id + "/" + args.bug_id + "/"
if os.path.exists(result_dir):
    shutil.rmtree(result_dir)
os.makedirs(result_dir)
shutil.copytree(report_dir, result_dir + "report")
