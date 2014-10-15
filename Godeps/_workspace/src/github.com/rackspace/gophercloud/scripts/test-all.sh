#!/bin/bash
#
# This script is responsible for executing all the acceptance tests found in
# the acceptance/ directory.

# Find where _this_ script is running from.
SCRIPTS=$(dirname $0)
SCRIPTS=$(cd $SCRIPTS; pwd)

# Locate the acceptance test / examples directory.
ACCEPTANCE=$(cd $SCRIPTS/../acceptance; pwd)

# Go workspace path
WS=$(cd $SCRIPTS/..; pwd)

# In order to run Go code interactively, we need the GOPATH environment
# to be set.
if [ "x$GOPATH" == "x" ]; then
  export GOPATH=$WS
  echo "WARNING: You didn't have your GOPATH environment variable set."
  echo "         I'm assuming $GOPATH as its value."
fi

# Run all acceptance tests sequentially.
# If any test fails, we fail fast.
LIBS=$(ls $ACCEPTANCE/lib*.go)
for T in $(ls -1 $ACCEPTANCE/[0-9][0-9]*.go); do
  if ! [ -x $T ]; then
    CMD="go run $T $LIBS -quiet"
    echo "$CMD ..."
    if ! $CMD ; then
      echo "- FAILED.  Try re-running w/out the -quiet option to see output."
      exit 1
    fi
  fi
done

