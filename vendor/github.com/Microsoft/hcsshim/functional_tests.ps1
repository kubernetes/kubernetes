# Requirements so far:
# dockerd running
#  - image microsoft/nanoserver (matching host base image)  docker load -i c:\baseimages\nanoserver.tar
#  - image alpine (linux) docker pull --platform=linux alpine


# TODO: Add this a parameter for debugging. ie "functional-tests -debug=$true"
#$env:HCSSHIM_FUNCTIONAL_TESTS_DEBUG="yes please"

#pushd uvm
go test -v -tags "functional uvmcreate uvmscratch uvmscsi uvmvpmem uvmvsmb uvmp9" ./...
#popd