#/usr/bin/env bash
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

SCRIPT_DIR=$TMP_DIR

LIB_DIR="/gzoltar/lib"
JUNIT_JAR="$LIB_DIR/junit.jar"
HAMCREST_JAR="$LIB_DIR/hamcrest-core.jar"

cd $SCRIPT_DIR

CLASS_DIR="$(defects4j export -p dir.bin.classes)"
TESTS_DIR="$(defects4j export -p dir.bin.tests)"
TARGET_DIR=$(echo $CLASS_DIR | sed -e "s/^\([^/]*\)\/.*/\1/g")
#TARGET_DIR="$TMP_DIR/target"

## Chartのみ
# mkdir "target"
# TARGET_DIR="$SCRIPT_DIR/target"
# cp -r $CLASS_DIR $TESTS_DIR $SCRIPT_DIR/target
# rm -r $CLASS_DIR $TESTS_DIR
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

echo "DONE!"
exit 0
