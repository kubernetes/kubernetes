#!/bin/bash
GITBRANCH=`git rev-parse --abbrev-ref HEAD`
#We intend to only run gas on release branches.
if [ "master" != $GITBRANCH ]; then
    exit 0
fi
REALEXITSTATUS=0
go get -u github.com/HewlettPackard/gas
gas -skip=*/arm/*/models.go -skip=*/management/examples/*.go -skip=*vendor* -skip=*/Gododir/* ./... | tee /dev/stderr
REALEXITSTATUS=$(($REALEXITSTATUS+$?))
gas -exclude=G101 ./arm/... ./management/examples/... | tee /dev/stderr
REALEXITSTATUS=$(($REALEXITSTATUS+$?))
gas -exclude=G204 ./Gododir/... | tee /dev/stderr
REALEXITSTATUS=$(($REALEXITSTATUS+$?))
exit $REALEXITSTATUS