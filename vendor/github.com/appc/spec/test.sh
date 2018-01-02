#!/usr/bin/env bash
#
# Run all appc tests
# ./test
# ./test -v
#
# Run tests for one package
#
# PKG=./discovery ./test
# PKG=schema/types ./test

set -e

# Invoke ./cover for HTML output
COVER=${COVER:-"-cover"}

source ./build.sh

TESTABLE_AND_FORMATTABLE="aci discovery pkg/acirenderer pkg/tarheader schema schema/lastditch schema/types"
FORMATTABLE="$TESTABLE_AND_FORMATTABLE ace actool"

# user has not provided PKG override
if [ -z "$PKG" ]; then
	TEST=$TESTABLE_AND_FORMATTABLE
	FMT=$FORMATTABLE

# user has provided PKG override
else
	# strip out leading dotslashes and trailing slashes from PKG=./foo/
	TEST=${PKG/#./}
	TEST=${TEST/#\//}
	TEST=${TEST/%\//}

	# only run gofmt on packages provided by user
	FMT="$TEST"
fi

# split TEST into an array and prepend REPO_PATH to each local package
split=(${TEST// / })
TEST=${split[@]/#/${REPO_PATH}/}

echo "Checking version..."
sver=$(bin/actool version|awk '{print $NF}')
read rver < VERSION
if [ "${sver}" != "${rver}" ]; then
	echo "schema/version.go and VERSION differ (${sver} != ${rver})"
	exit 255
fi

echo "Running tests..."
go test -timeout 60s ${COVER} $@ ${TEST} --race

echo "Validating image manifest..."
bin/actool validate examples/image.json

echo "Validating pod template manifest..."
bin/actool validate examples/pod_template.json

# TODO(jonboulle): add `actool validate --resolved-pod-manifest`
echo "Validating pod runtime manifest..."
bin/actool validate examples/pod_runtime.json

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

echo "Success"
