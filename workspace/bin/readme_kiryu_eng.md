
# Overview

This is a tool for collecting defects4j coverage using Jacoco.
The process of collecting coverage is as follows:

1. Compile defects4j projects.
2. Embed code in class files.
3. Perform each test and collect the execution logs of each test.
    - The tests to be executed are aggregated with gzoltar (saved in all_tests.txt).
    - An exec file with the same name as each test is collected.
4. Extract coverage from execution logs.

# Usage

1. Build services with docker-compose.
2. Enter the service and run setup.sh (/workspace/bin/setup.sh, probably accessible via the path).
3. run_test.sh <project_id> <bug_id>
4. /workspace/clover/<project_id>/<bug_id>

# Each file in workspace/bin

- checkout-compile.sh
    - Performs steps 1 and 2 of the process, and test aggregation by gzoltar.
    - Provide the project name and bug id as arguments (e.g., "checkout-compile.sh Chart 2").
    - The compiled and embedded project is generated in "/tmp/project name/bug id".
    - Embedded class files (under instrumented) are generated from normal class files after compilation. Executing this generates the execution logs.
- collect_coverage_parallelized.py
    - Performs steps 3 and 4.
    - Provide the project name and bug id as arguments (e.g., "collect_coverage_parallelized.py Chart 2").
    - Parallel execution of tests aggregated by gzoltar saved in all_tests.txt.
    - Coverage is generated in "workspace/clover/project name/bug id".
        - The coverage file name for failed tests starts with "fail".
        - "is_plane.txt" is generated, and 1 is attached to the parts without statements in the source code. "is_plane.txt" collects the parts with 0 (lines with statements) as coverage. Therefore, the line numbers of coverage files do not match the line numbers of the source code. However, in test.py, it reads the coverage considering this, so the line numbers match.
    - Although it remains as a program, as can be seen by reading run_test.sh, it is not used when using OpenClover.
- hoge.py
    - Program to collect execution count coverage information. Used by default in run_test.sh.
- hoge2.py
    - Program to collect regular code coverage information. Uncomment it if you want to use it. In that case, comment out hoge.py and hoge3.py.
- hoge3.py
    - Program to collect execution traces used in the papers [Fault Localization with DNN-based Test Case Learning and Ablated Execution Traces](../../ISE2023-ikeda_t.pdf). Uncomment it when used, similar to hoge.py, comment out hoge2.py.
- run_test.sh
    - Runs checkout-compile.sh and collect_coverage_parallelized.py.
    - Provide the project name and bug id as arguments.
    
    ```
    run_test.sh Lang 1
    
    ```
    
    - By default, it is set to use OpenClover. Basically, collect coverage information using OpenClover. This is because in some programs, code coverage information cannot be collected with Jacoco.
    - Since dependency information varies for each project, there are some parts that need to be changed for each version of the project in several programs and shell scripts. The following files are files that need to be changed. Identify the necessary places by utilizing the search function of the editor.
        - run_test.sh
        - hoge.py (hoge2.py does not need to be changed because it omits the test execution function. However, coverage reports collected by hoge.py are required for execution.)
        - hoge3.py
        - writemavenxml.py
        - xml_write_test.py
- SingleJUnitTestRunner.java
    - Since I didn't know how to run JUnit tests individually, I made a class that forcibly runs individual test functions. collect_coverage_parallelized.py executes each test through this class.
    - If ant or JUnit itself has a function to run tests individually, this is unnecessary.

# Files in ml

- test.py
    - Machine learning part
    - The version of TensorFlow was 2.7.0
    - Please rewrite the part of os.path.join
- bugs_to_be_collected.txt
    - Target bugs
- chunks
    - Fixed parts of each bug (line number and chunk set)
    - Scraped from "[http://program-repair.org/defects4j-dissection/#!/](http://program-repair.org/defects4j-dissection/#!/)"

# How to use Clover

1. Embed
    
    ```
    java -cp /root/clover/lib/clover.jar com.atlassian.clover.CloverInstr --source <Java version of source file (somewhere in the configuration file of Maven or Ant)> -i clover.db -s Source directory before embedding -d Source directory after embedding
    
    ```
    
    With the above code, you can create source files with embedded collected code. It's easy to compile by setting the source directory src to tmp_src first, then compiling with -s tmp_src -d src. The source directory depends on the project, but you can get it with "defects4j export -p dir.src.tests".
    
2. Compile
Compile with defects4j compile.
3. Test aggregation
Aggregate with gzoltar at the end of checkout-compile.sh
4. Execution
    
    ```
    java -cp /root/clover/lib/clover.jar:/workspace/bin:$(defects4j export -p cp.test) SingleJUnitTestRunner Test item (corresponding to each line of all_tests.txt)
    
    ```
    
    Include clover.jar, SingleJunitTestRunner, and things necessary for the project tests in the classpath.
    
5. Extract coverage report
    
    ```
    java -cp /root/clover/lib/clover.jar com.atlassian.clover.reporters.xml.XMLReporter -if -l -i clover.db -o Coverage report file
    
    ```
    
6. Generate coverage
Use the latter half of collect_coverage_parallelized.py to generate coverage from the coverage report.

# Points to note

- The directory structure varies considerably depending on the project to be collected (and even within the same project depending on the bug id), so it may be necessary to match the paths of class files in checkout-compile.sh and collect_coverage_parallelized.py for each project.
- The method using clover is not automated (too dependent on the project), so it needs to be done manually.
- Since clover does not generate an exec file for each test but saves the accumulated execution logs in clover.db, "Extract reports from clover.db and reset clover.db (copy the clover.db generated during embedding as tmp_clover.db and copy it to clover.db each time)" is required every time the tests are run. Therefore, parallel test execution is not possible, and it takes a long time.