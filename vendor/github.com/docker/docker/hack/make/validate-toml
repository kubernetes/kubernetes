#!/bin/bash

source "${MAKEDIR}/.validate"

IFS=$'\n'
files=( $(validate_diff --diff-filter=ACMR --name-only -- 'MAINTAINERS' || true) )
unset IFS

badFiles=()
for f in "${files[@]}"; do
	# we use "git show" here to validate that what's committed is formatted
	if ! git show "$VALIDATE_HEAD:$f" | tomlv /proc/self/fd/0 ; then
		badFiles+=( "$f" )
	fi
done

if [ ${#badFiles[@]} -eq 0 ]; then
	echo 'Congratulations!  All toml source files changed here have valid syntax.'
else
	{
		echo "These files are not valid toml:"
		for f in "${badFiles[@]}"; do
			echo " - $f"
		done
		echo
		echo 'Please reformat the above files as valid toml'
		echo
	} >&2
	false
fi
