#!/bin/bash
set -o errexit

cd $(dirname $0)
ls $GOPATH/src/github.com/cloudflare/cfssl

PACKAGES=$(go list ./... | grep -v /vendor/)

go vet $PACKAGES
if ! which fgt > /dev/null ; then
    echo "Please install fgt from https://github.com/GeertJohan/fgt."
    exit 1
fi

if ! which golint > /dev/null ; then
    echo "Please install golint from github.com/golang/lint/golint."
    exit 1
fi

for package in $PACKAGES
do
    fgt golint ${package}
done

# check go fmt
for package in $PACKAGES
do
    test -z "$(gofmt -s -l $GOPATH/src/$package/ | tee /dev/stderr)"
done

# Build and install cfssl executable in PATH
go install -tags "$BUILD_TAGS" github.com/cloudflare/cfssl/cmd/cfssl

COVPROFILES=""
for package in $(go list -f '{{if len .TestGoFiles}}{{.ImportPath}}{{end}}' $PACKAGES)
do
    profile="$GOPATH/src/$package/.coverprofile"
    go test -tags "$BUILD_TAGS" --coverprofile=$profile $package
    [ -s $profile ] && COVPROFILES="$COVPROFILES $profile"
done
cat $COVPROFILES > coverprofile.txt
