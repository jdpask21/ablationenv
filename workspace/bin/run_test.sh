die() {
  echo -e "$@" >&2
  exit 1
}
USAGE="Usage: $0 <pid> <bid>\n\
pid: Defects4J's Project ID which is like Chart, Math, ...\n\
bid: Defects4J's Bug  ID \n"
[ $# -eq 2 ] || die "$USAGE"
PID="$1"
BID="$2"


/workspace/bin/checkout-compile.sh $PID $BID || die "\e[31mfailed at checkout-compile $PID-$BID\e[m"
echo "checked out $PID-$BID"
# sleep 2
python -u /workspace/bin/collect_coverage_parallelized.py $PID $BID #|| die "failed at collect_coverage $PID-$BID"
echo "collected $PID-$BID"
# sleep 8

# # > /dev/null 2>&1
