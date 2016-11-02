# Requirements

## Go 1.5 or later

Follow [the instructions](https://golang.org/doc/install) to install Go. Once
the installation is complete, run:

    go version

to verify that you have installed the version 1.5 or later. Then, create your
[workspace](https://golang.org/doc/code.html#Workspaces) and set the `GOPATH`
and environment variable to point to your workspace:

    mkdir $HOME/go
    export GOPATH=$HOME/go

Add `$GOPATH/bin` to your `PATH`:

    export PATH=$PATH:$GOPATH/bin

## Godep

Run:

    go get github.com/tools/godep

and verify that the `godep` command is installed in `$GOPATH/bin`:

# Checking out the source and building

Clone the `photon-controller-go-sdk` repo under `$GOPATH/src/github.com/vmware`:

    mkdir -p $GOPATH/src/github.com/vmware
    cd $GOPATH/src/github.com/vmware
    git clone (github.com/vmware or gerrit)/photon-controller-go-sdk

Then, restore dependencies and install the `ginkgo` command:

    cd $GOPATH/src/github.com/vmware/photon-controller-go-sdk
    godep restore
    go install github.com/onsi/ginkgo/ginkgo

and verify that the `ginkgo` command is installed in `$GOPATH/bin`:

    ls $GOPATH/bin/ginkgo

To compile, run:

    go build ./...

# Testing

Run:

    go test ./... -v

to run the tests against a mock server. You can set `TEST_ENDPOINT` environment
variable to run the tests against a real Photon Controller endpoint:

    # Need a longer timeout against a real server
    TEST_ENDPOINT=http://localhost:9080 go test ./... -v -timeout 1800s

With `ginkgo`, you can run a subset of the tests:

    TEST_ENDPOINT=http://localhost:9080 ginkgo -r -focus Tenant -v
