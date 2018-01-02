#!/usr/bin/env bash
#
# Run all CNI tests
#   ./test
#   ./test -v
#
# Run tests for one package
#   PKG=./libcni ./test
#
set -e

source ./build.sh

TESTABLE="libcni pkg/invoke pkg/skel pkg/types pkg/types/current pkg/types/020 pkg/version pkg/version/testhelpers plugins/test/noop"
FORMATTABLE="$TESTABLE"

# user has not provided PKG override
if [ -z "$PKG" ]; then
	TEST=$TESTABLE
	FMT=$FORMATTABLE

# user has provided PKG override
else
	# strip out slashes and dots from PKG=./foo/
	TEST=${PKG//\//}
	TEST=${TEST//./}

	# only run gofmt on packages provided by user
	FMT="$TEST"
fi

# split TEST into an array and prepend REPO_PATH to each local package
split=(${TEST// / })
TEST=${split[@]/#/${REPO_PATH}/}

echo -n "Running tests "
function testrun {
    sudo -E bash -c "umask 0; PATH=$GOROOT/bin:$(pwd)/bin:$PATH go test -covermode set $@"
}
if [ ! -z "${COVERALLS}" ]; then
    echo "with coverage profile generation..."
    i=0
    for t in ${TEST}; do
        testrun "-coverprofile ${i}.coverprofile ${t}"
        i=$((i+1))
    done
    gover
    goveralls -service=travis-ci -coverprofile=gover.coverprofile -repotoken=$COVERALLS_TOKEN
else
    echo "without coverage profile generation..."
    testrun "${TEST}"
fi

echo "Checking gofmt..."
fmtRes=$(gofmt -l $FMT)
if [ -n "${fmtRes}" ]; then
	echo -e "gofmt checking failed:\n${fmtRes}"
	exit 255
fi

echo "Checking govet..."
vetRes=$(go vet $TEST)
if [ -n "${vetRes}" ]; then
	echo -e "govet checking failed:\n${vetRes}"
	exit 255
fi

echo "Checking license header..."
licRes=$(
       for file in $(find . -type f -iname '*.go' ! -path './vendor/*'); do
               head -n1 "${file}" | grep -Eq "(Copyright|generated)" || echo -e "  ${file}"
       done
)
if [ -n "${licRes}" ]; then
       echo -e "license header checking failed:\n${licRes}"
       exit 255
fi


echo "Success"
