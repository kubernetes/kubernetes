#!/bin/sh

# We cannot use go test -coverprofile=cover.out ./... because
# the tool requires that it be used only on one package when
# capturing the coverage
# This is why we need this little script here.
packages="./apps/glusterfs"
packages="${packages} ./executors/sshexec"
packages="${packages} ./executors/mockexec"
packages="${packages} ./executors/kubeexec"
packages="${packages} ./client/api/go-client"
packages="${packages} ./middleware"
packages="${packages} ./pkg/utils"
COVERFILE=packagecover.out

coverage()
{

    echo "mode: count" > $COVERFILE
    for pkg in $packages ; do
        echo "-- Testing $pkg --"

        # Collect coverage
        go test -covermode=count -coverprofile=cover.out $pkg || exit 1

        # Show in the command line
        go tool cover -func=cover.out

        # Append to coverfile
        grep -v "^mode: count" cover.out >> $COVERFILE

        # Cleanup
        rm -f cover.out
    done
}

coverage

if [ -n "$COVERALLS_TOKEN" ] ; then
    # Send to coveralls.io
    $HOME/gopath/bin/goveralls \
        -coverprofile=$COVERFILE \
        -service=travis-ci \
        -repotoken $COVERALLS_TOKEN
fi

# Clean up
rm -f $COVERFILE > /dev/null 2>&1
exit 0
