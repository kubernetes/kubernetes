#!/bin/bash
set -e

: ${PARALLEL_JOBS:=$(nproc 2>/dev/null || echo 1)} # if nproc fails (usually because we don't have it), let's not parallelize by default

RED=$'\033[31m'
GREEN=$'\033[32m'
TEXTRESET=$'\033[0m' # reset the foreground colour

# Run Docker's test suite, including sub-packages, and store their output as a bundle
# If $TESTFLAGS is set in the environment, it is passed as extra arguments to 'go test'.
# You can use this to select certain tests to run, eg.
#
#   TESTFLAGS='-test.run ^TestBuild$' ./hack/make.sh test-unit
#
bundle_test_unit() {
	{
		date

		# Run all the tests if no TESTDIRS were specified.
		if [ -z "$TESTDIRS" ]; then
			TESTDIRS=$(find_dirs '*_test.go')
		fi
		(
			export LDFLAGS
			export TESTFLAGS
			export HAVE_GO_TEST_COVER

			# some hack to export array variables
			export BUILDFLAGS_FILE="$DEST/buildflags-file"
			( IFS=$'\n'; echo "${BUILDFLAGS[*]}" ) > "$BUILDFLAGS_FILE"

			if command -v parallel &> /dev/null; then
				# accomodate parallel to be able to access variables
				export SHELL="$BASH"
				export HOME="$(mktemp -d)"
				mkdir -p "$HOME/.parallel"
				touch "$HOME/.parallel/ignored_vars"

				echo "$TESTDIRS" | parallel --jobs "$PARALLEL_JOBS" --env _ "${MAKEDIR}/.go-compile-test-dir"
				rm -rf "$HOME"
			else
				# aww, no "parallel" available - fall back to boring
				for test_dir in $TESTDIRS; do
					"${MAKEDIR}/.go-compile-test-dir" "$test_dir" || true
					# don't let one directory that fails to build tank _all_ our tests!
				done
			fi
			rm -f "$BUILDFLAGS_FILE"
		)
		echo "$TESTDIRS" | go_run_test_dir
	}
}

go_run_test_dir() {
	TESTS_FAILED=()
	while read dir; do
		echo
		echo '+ go test' $TESTFLAGS "${DOCKER_PKG}${dir#.}"
		precompiled="$ABS_DEST/precompiled/$dir.test$(binary_extension)"
		if ! ( cd "$dir" && test_env "$precompiled" $TESTFLAGS ); then
			TESTS_FAILED+=("$dir")
			echo
			echo "${RED}Tests failed: $dir${TEXTRESET}"
			sleep 1 # give it a second, so observers watching can take note
		fi
	done

	echo
	echo
	echo

	# if some tests fail, we want the bundlescript to fail, but we want to
	# try running ALL the tests first, hence TESTS_FAILED
	if [ "${#TESTS_FAILED[@]}" -gt 0 ]; then
		echo "${RED}Test failures in: ${TESTS_FAILED[@]}${TEXTRESET}"
		echo
		false
	else
		echo "${GREEN}Test success${TEXTRESET}"
		echo
		true
	fi
}

bundle_test_unit 2>&1 | tee -a "$DEST/test.log"
