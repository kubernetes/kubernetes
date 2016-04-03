#!/bin/bash
set -e

bundle_cover() {
	coverprofiles=( "$DEST/../"*"/coverprofiles/"* )
	for p in "${coverprofiles[@]}"; do
		echo
		(
			set -x
			go tool cover -func="$p"
		)
	done
}

if [ "$HAVE_GO_TEST_COVER" ]; then
	bundle_cover 2>&1 | tee "$DEST/report.log"
else
	echo >&2 'warning: the current version of go does not support -cover'
	echo >&2 '  skipping test coverage report'
fi
