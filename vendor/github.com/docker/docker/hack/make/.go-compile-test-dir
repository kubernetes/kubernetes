#!/bin/bash
set -e

# Compile phase run by parallel in test-unit. No support for coverpkg

dir=$1
in_file="$dir/$(basename "$dir").test"
out_file="$DEST/precompiled/$dir.test"
# we want to use binary_extension() here, but we can't because it's in main.sh and this file gets re-execed
if [ "$(go env GOOS)" = 'windows' ]; then
	in_file+='.exe'
	out_file+='.exe'
fi
testcover=()
if [ "$HAVE_GO_TEST_COVER" ]; then
	# if our current go install has -cover, we want to use it :)
	mkdir -p "$DEST/coverprofiles"
	coverprofile="docker${dir#.}"
	coverprofile="$DEST/coverprofiles/${coverprofile//\//-}"
	testcover=( -cover -coverprofile "$coverprofile" ) # missing $coverpkg
fi
if [ "$BUILDFLAGS_FILE" ]; then
	readarray -t BUILDFLAGS < "$BUILDFLAGS_FILE"
fi

if ! (
	cd "$dir"
	go test "${testcover[@]}" -ldflags "$LDFLAGS" "${BUILDFLAGS[@]}" $TESTFLAGS -c
); then
	exit 1
fi

mkdir -p "$(dirname "$out_file")"
mv "$in_file" "$out_file"
echo "Precompiled: ${DOCKER_PKG}${dir#.}"
