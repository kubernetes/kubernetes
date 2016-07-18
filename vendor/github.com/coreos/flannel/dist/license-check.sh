#!/usr/bin/env bash

licRes=$(for file in $(find . -type f -iname '*.go' ! -path './vendor/*'); do
		head -n3 "${file}" | grep -Eq "(Copyright|generated|GENERATED)" || echo -e "  ${file}"
	done;)
if [ -n "${licRes}" ]; then
	echo -e "license header checking failed:\n${licRes}"
	exit 255
fi