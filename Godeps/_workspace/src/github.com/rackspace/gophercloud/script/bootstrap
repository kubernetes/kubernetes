#!/bin/bash
#
# This script helps new contributors set up their local workstation for
# gophercloud development and contributions.

# Create the environment
export GOPATH=$HOME/go/gophercloud
mkdir -p $GOPATH

# Download gophercloud into that environment
go get github.com/rackspace/gophercloud
cd $GOPATH/src/github.com/rackspace/gophercloud
git checkout master

# Write out the env.sh convenience file.
cd $GOPATH
cat <<EOF >env.sh
#!/bin/bash
export GOPATH=$(pwd)
export GOPHERCLOUD=$GOPATH/src/github.com/rackspace/gophercloud
EOF
chmod a+x env.sh

# Make changes immediately available as a convenience.
. ./env.sh

