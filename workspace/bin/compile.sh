SCRIPT_DIR=$TMP_DIR

LIB_DIR="/gzoltar/lib"
JUNIT_JAR="$LIB_DIR/junit.jar"
HAMCREST_JAR="$LIB_DIR/hamcrest-core.jar"


#
# Compile
#

echo "Compile source and test cases ..."

cd $SCRIPT_DIR
# エンコーディングで文句言われがち
# for xml in $(find -name "*build.xml"); do sed -i -e "s/<javac/<javac encoding=\"utf-8\"/g" ${xml}; done
defects4j compile || die "compiling the project failed."
