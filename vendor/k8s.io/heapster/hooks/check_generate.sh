#!/bin/bash

# Copyright 2015 Google Inc. All rights reserved.
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

GEN_FILES="extpoints/extpoints.go"

md5sums=""

for f in $GEN_FILES; do
	md5sums="${md5sums}$(md5sum $f)\n"
done

go generate 2>/dev/null || exit 1

echo -ne "$md5sums" | while read sum f; do
	newsum=$(md5sum $f | awk '{ print $1 }')
	if [[ ! $newsum == $sum ]]; then
		echo "go generate was not run, $f needs to be generated again"
		exit 1
	fi
done
