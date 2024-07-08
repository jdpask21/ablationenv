#!/usr/bin/env bash
# ------------------------------------------------------------------------------
#
#
# Usage:
# ./main.sh <pid> <bid>
#
# Parameters:
# - <pid>  Defects4J's project ID: Chart, Closure, Lang, Math, Mockito or Time.
# - <bid>  Bug ID
#
# Examples:
#   $ ./main.sh Chart 5
#
# Environment variables:
# - ANT_HOME   Needs to be set and must point to the Ant installation.
# - D4J_HOME   Needs to be set and must point to the Defects4J installation.

# ------------------------------------------------------------------------------

die() {
  echo -e "$@" >&2
  exit 1
}
# # Check whether JAVA_HOME is set
# [ "$JAVA_HOME" != "" ] || die "[ERROR] JAVA_HOME is not set!"
# [ -d "$JAVA_HOME" ] || die "$JAVA_HOME does not exist!"
# # Check whether ANT_HOME is set
# [ "$ANT_HOME" != "" ] || die "[ERROR] ANT_HOME is not set!"
# [ -d "$ANT_HOME" ] || die "$ANT_HOME does not exist!"
# # Check whether D4J_HOME is set
# [ "$D4J_HOME" != "" ] || die "[ERROR] D4J_HOME is not set!"
# [ -d "$D4J_HOME" ] || die "$D4J_HOME does not exist!"

export PATH="$JAVA_HOME/bin:$ANT_HOME/bin:$PATH"

USAGE="Usage: $0 <pid> <bid>\n\
pid: Defects4J's Project ID which is like Chart, Math, ...\n\
bid: Defects4J's Bug  ID \n"
[ $# -eq 2 ] || die "$USAGE"
PID="$1"
BID="$2"

TMP_DIR="/tmp/$PID/$BID"
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"


echo "PID: $$"
hostname
java -version

echo ""
echo "[INFO] Checkout $PID-${BID}_buggy"
defects4j checkout -p "$PID" -v "$BID""b" -w "$TMP_DIR"
if [ $? -ne 0 ]; then
  echo "[ERROR] Checkout of the $PID-${BID}b version has failed!"
  rm -rf "$TMP_DIR"
  exit 1
fi

SCRIPT_DIR=$TMP_DIR

LIB_DIR="/gzoltar/lib"
JUNIT_JAR="$LIB_DIR/junit.jar"
HAMCREST_JAR="$LIB_DIR/hamcrest-core.jar"

cd $SCRIPT_DIR

###Math version -82?
#cp -r src/test/java src/test/tmp_java
#cp -r src/main/java src/main/tmp_java
###Math version 85-? 
#cp -r src/test src/tmp_test
#cp -r src/java src/tmp_java
####Chart
cp -r tests tmp_tests
cp -r source tmp_source
####Closure
#cp -r test tmp_test
#cp -r src tmp_src
###Lang version -35
# cp -r src/test/java src/test/tmp_java
# cp -r src/main/java src/main/tmp_java
###Lang version 36-
#cp -r src/test src/tmp_test
#cp -r src/java src/tmp_java

export JAVA_TOOL_OPTIONS=-Dfile.encoding=UTF8   ###one time is OK

###Math version -82?
#java -cp /root/clover/lib/clover.jar com.atlassian.clover.CloverInstr --source 8 -i clover.db -s src/main/tmp_java -d src/main/java
###Math version 85-? 
#java -cp /root/clover/lib/clover.jar com.atlassian.clover.CloverInstr --source 1.5 -i clover.db -s src/tmp_java -d src/java
###Chart
java -cp /root/clover/lib/clover.jar com.atlassian.clover.CloverInstr --source 1.3 -i clover.db -s tmp_source -d source
###Closure
#java -cp /root/clover/lib/clover.jar com.atlassian.clover.CloverInstr --source 1.5 -i clover.db -s tmp_src -d src
###Lang version -35
# java -cp /root/clover/lib/clover.jar com.atlassian.clover.CloverInstr --source 8 -i clover.db -s src/main/tmp_java -d src/main/java
###Lang version 36-
#java -cp /root/clover/lib/clover.jar com.atlassian.clover.CloverInstr --source 8 -i clover.db -s src/tmp_java -d src/java
###Lang version 46-
#java -cp /root/clover/lib/clover.jar com.atlassian.clover.CloverInstr --source 1.3 -i clover.db -s src/tmp_java -d src/java

cp clover.db tmp_clover.db
#mkdir $SCRIPT_DIR/target
#mkdir $SCRIPT_DIR/target/lib
#mkdir $SCRIPT_DIR/target/lib/lib
#cp /root/clover/lib/clover.jar $SCRIPT_DIR/target/lib/lib/

#cp ../85/build.xml ./build.xml   ##Math 94-
python /workspace/bin/xml_write_test.py $PID $BID   ###基本これ
# python /workspace/bin/writemavenxml.py $PID $BID   ###Lang from Version21 to 41

defects4j compile  || die "compiling the project failed."

CLASS_DIR="$(defects4j export -p dir.bin.classes)"
TESTS_DIR="$(defects4j export -p dir.bin.tests)"
TARGET_DIR=$(echo $CLASS_DIR | sed -e "s/^\([^/]*\)\/.*/\1/g")
#TARGET_DIR="$TMP_DIR/target"

## Chartのみ
mkdir "target"
TARGET_DIR="$SCRIPT_DIR/target"
cp -r $CLASS_DIR $TESTS_DIR $SCRIPT_DIR/target

# rm -r $CLASS_DIR $TESTS_DIR   ###コメントアウトしないと動作しない。なにこれ？
# CLASS_DIR=$TARGET_DIR/$CLASS_DIR
# TESTS_DIR=$TARGET_DIR/$TESTS_DIR

#
# jacococli instrument for buggy class
#
#INSTRUMENTED_DIR="$SCRIPT_DIR/instrumented"
#cp -r $TARGET_DIR $INSTRUMENTED_DIR

#
# Collect list of unit test cases to run
#

UNIT_TESTS_FILE="$SCRIPT_DIR/all_tests.txt"
java -cp $CLASS_DIR:$TESTS_DIR:$JUNIT_JAR:$HAMCREST_JAR:$GZOLTAR_CLI_JAR \
  com.gzoltar.cli.Main listTestMethods $TARGET_DIR \
    --outputFile "$UNIT_TESTS_FILE" || die "Collection of unit test cases has failed!"
[ -s "$UNIT_TESTS_FILE" ] || die "$UNIT_TESTS_FILE does not exist or it is empty!"

python -u /workspace/bin/hoge.py $PID $BID   ###実行回数カバレッジ

# python -u /workspace/bin/hoge2.py $PID $BID   ###通常のコードカバレッジ

# python -u /workspace/bin/hoge3.py -c $PID $BID   ###実行回数カバレッジ -cオプションはカバレッジレポートがあるが、カバレッジ情報のみ収集したい際に使用する。テストの実行が行われないオプション。



echo "DONE!"
exit 0
