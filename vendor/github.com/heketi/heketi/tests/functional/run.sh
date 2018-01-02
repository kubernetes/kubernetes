#!/bin/sh

source ./lib.sh

teardown_all() {
    results=0
    for testDir in * ; do
        if [ -x $testDir/teardown.sh ] ; then
            println "TEARDOWN $testDir"
            cd $testDir
            teardown.sh
            cd ..
        fi
    done
}

### MAIN ###

# See https://bugzilla.redhat.com/show_bug.cgi?id=1327740
_sudo setenforce 0

starttime=`date`
export PATH=$PATH:.

# Check go can build
if [ -z $GOPATH ] ; then
    fail "GOPATH must be specified"
fi

# Clean up
rm -f heketi-server > /dev/null 2>&1
teardown_all

# Check each dir for tests
results=0
for testDir in * ; do
    if [ -x $testDir/run.sh ] ; then
        println "TEST $testDir"
        cd $testDir

        # Run the command with a large timeout.
        # Just large enough so that it doesn't run forever.
        timeout 1h run.sh ; result=$?

        if [ $result -ne 0 ] ; then
            println "FAILED $testDir"
            println "TEARDOWN $testDir"
            teardown.sh
            results=1
        else
            println "PASSED $testDir"
        fi

        cd ..
    fi
done

# Summary
println "Started $starttime"
println "Ended `date`"
if [ $results -eq 0 ] ; then
    println "PASSED"
else
    println "FAILED"
fi

exit $results
