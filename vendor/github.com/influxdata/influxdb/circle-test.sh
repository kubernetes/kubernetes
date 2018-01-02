#!/bin/bash
#
# This is the InfluxDB test script for CircleCI, it is a light wrapper around ./test.sh.

# Exit if any command fails
set -e

# Get dir of script and make it is our working directory.
DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
cd $DIR

export OUTPUT_DIR="$CIRCLE_ARTIFACTS" 
# Don't delete the container since CircleCI doesn't have permission to do so.
export DOCKER_RM="false"

# Get number of test environments.
count=$(./test.sh count)
# Check that we aren't wasting CircleCI nodes.
if [ $CIRCLE_NODE_TOTAL -gt $count ]
then
    echo "More CircleCI nodes allocated than tests environments to run!"
    exit 1
fi

# Map CircleCI nodes to test environments.
tests=$(seq 0 $((count - 1)))
for i in $tests
do
    mine=$(( $i % $CIRCLE_NODE_TOTAL ))
    if [ $mine -eq $CIRCLE_NODE_INDEX ]
    then
        echo "Running test env index: $i"
        ./test.sh $i
    fi
done
