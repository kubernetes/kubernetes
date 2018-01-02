#!/bin/sh

CURRENT_DIR=`pwd`

# Build server if we need to
if [ ! -x heketi-server ] ; then
    ( cd ../../../ ; make ; cp heketi $CURRENT_DIR/heketi-server )
fi

# Start server
rm -f heketi.db > /dev/null 2>&1
./heketi-server --config=test/unit/heketi.json > heketi.log 2>&1 &
pid=$!
sleep 2

# Start unit tests
tox -e py27
results=$?


# kill server
kill $pid

if [ $results -ne 0 ] ; then
    exit $results
fi

tox -e pep8
results=$?

exit $results
