#!/bin/bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

if [[ -z ${GO_MD2MAN} ]]; then
	GO_MD2MAN="go-md2man"
fi

# get into this script's directory
cd "$(dirname "$(readlink -f "$BASH_SOURCE")")"

[ "$1" = '-q' ] || {
	set -x
	pwd
}

for FILE in *.md; do
	base="$(basename "$FILE")"
	name="${base%.md}"
	num="${name##*.}"
	if [ -z "$num" -o "$name" = "$num" ]; then
		# skip files that aren't of the format xxxx.N.md (like README.md)
		continue
	fi
	mkdir -p "./man${num}"
	${GO_MD2MAN} -in "$FILE" -out "./man${num}/${name}"
done
