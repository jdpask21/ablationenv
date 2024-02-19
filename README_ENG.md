## About the contents of each directory and file

- fl
<br>Contains Docker files.
- workspace
<br>
    - bin<br>
    Contains programs and experimental data used in experiments.
        - ablationex_util<br>
        Contains common functions used in programs for experiments.
        - images-susscore<br>
        Contains results of applying the main research and comparison methods to the experimental program.
        - line_sfl_data<br>
        Contains experimental results.
        - seq_sus_{ab,ochiai,tara}<br>
        Contains data of suspicion scores for each line of each program. This is the result data of the experiments.
    - chunks<br>
    Contains chunks indicating the line numbers of bugs in each program.
    - clover-line<br>
    Contains execution traces used in this research. The same traces can be collected by referring to Mr. Kiryu's handover document.

## About the programs in /workspace/bin

Non-essential programs are also retained, but explanations are omitted.

- cal_ochiai.py<br>
Program for executing Ochiai and Tarantula.
- cal_bugpattern.py<br>
Program used to consider experimental results by fault. It is retained, although it is not likely to be used.
- checkout.sh, compile.sh, run_checkout.sh, run_test.sh, setup.sh, zoltar.sh, collect_coverage_parallelized.py, hoge.py, hoge2.py, hoge3.py, coverageout.py, writemavenxml.py, xml_write_test.py, SIngleJUnitTestRUnner.class<br>
Shell and programs for collecting execution traces and coverage. For details, please refer to the [handover document](./workspace/bin/readme_kiryu_eng.md).
- collect_bug_multi.txt, collect{PROJECT_NAME}.txt<br>
Text files containing projects of multiple faults and the names and versions of each project.
- main_line.py<br>
Program for training and saving DNN models in this paper. By default, it is set to use all data as training data. See below for usage.
- onlin_line.py<br>
Program to calculate suspicion scores using the proposed method in this research. Requires the DNN model created by main_line.py for use. See below for usage.
- plot_susscore.py<br>
Program to plot suspicion scores output by onlin_line.py. Used for analysis, etc.
- rename_model.py<br>
Program to change the names of saved DNN models to facilitate experimentation when using multiple models and averaging them. This program modifies the names of models saved by main_line.py so that they can be processed collectively by onlin_line.py.
- make_linetr.py<br>
Program to collect execution traces used in this research from coverage reports collected using gcov. It was created for SIR, so it cannot be used for Defects4j coverage reports. For Defects4j, since it is in xml format, the same traces can be collected by writing a program.

## Building the experimental environment and running programs

1. Create a Docker image and start the container in the directory with README.

```
docker-compose up -d

```

1. Enter the started container and move to /workspace/bin.

```
docker exec -it [CONTAINER_ID] bash
cd /workspace/bin

```

1. To train and save the DNN model, execute main_line.py. When executing, you need to provide the project name and version as variables. The example below trains the model for totinfo version 2. Depending on the project, training the model may take some time (about 30 minutes to 1 hour?). Also, by default, all data is set to be used as training data, and if you want to change the training data size, edit the following global variables.
    - TRAINING_FULL_DATA = False
    - TRAINING_SIZE_ = [Any float value (in the range 0~1)]
    
    Also, the program uses an integer value stored in the variable SEED_VALUE as the directory name for saving the model, and if a directory with the same value already exists, the model will not be trained and the process will end. In that case, specify a different value or delete the existing directory.
    

```
python3 main_line.py totinfo 2

```

1. To calculate suspicion scores using the trained DNN model, execute onlin_line.py. onlin_line.py also uses the variable SEED_VALUE to specify which DNN model to use. Please set it to an appropriate value. Also, even with the same SEED value, there may be multiple models saved, so use the variable NNs_MODEL_NAME to specify which model to use. Change this to an appropriate name as well. In the experiments of this paper, we basically use the first epoch model that meets the saving conditions of the DNN model (precision, recall, TNR>=0.8).

```
python3 onlin_line.py totinfo 2

```

1. Confirm the results. In the paper, 10 models were created for each program, so the average of the results of each model is taken as the final experimental result. The average result is written to the file specified by WRITE_TOPN_PERCENTILE, and the results of each model are written to the file specified by OUTPUT_PER_SEED. The written result is the TopN% of code investigation required to identify faults.