#!/bin/bash

# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

dirs=$(find . -type d -not -path "*/internal*" -not -path "*/test*" -not -path "*/example*" -not -path "*/.*" | sort)

missingReadme=false
for dir in $dirs; do
	if [ ! -f "$dir/README.md" ]; then
		echo "couldn't find README.md for $dir"
		missingReadme=true
	fi
done

if [ "$missingReadme" = true ] ; then
	echo "Error: some READMEs couldn't be found."
	exit 1
fi
