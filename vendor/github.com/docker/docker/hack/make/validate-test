#!/bin/bash

# Make sure we're not using gos' Testing package any more in integration-cli

source "${MAKEDIR}/.validate"

IFS=$'\n'
files=( $(validate_diff --diff-filter=ACMR --name-only -- 'integration-cli/*.go' || true) )
unset IFS

badFiles=()
for f in "${files[@]}"; do
	# skip check_test.go since it *does* use the testing package
	if [ "$f" = "integration-cli/check_test.go" ]; then
		continue
	fi

	# we use "git show" here to validate that what's committed is formatted
	if git show "$VALIDATE_HEAD:$f" | grep -q testing.T; then
		badFiles+=( "$f" )
	fi
done

if [ ${#badFiles[@]} -eq 0 ]; then
	echo 'Congratulations! No testing.T found.'
else
	{
		echo "These files use the wrong testing infrastructure:"
		for f in "${badFiles[@]}"; do
			echo " - $f"
		done
		echo
	} >&2
	false
fi
