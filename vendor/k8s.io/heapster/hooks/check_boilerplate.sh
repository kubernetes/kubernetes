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

REF_FILE="./hooks/boilerplate.go.txt"
if [ ! -e $REF_FILE ]; then
  echo "Missing reference file: " ${REF_FILE}
  exit 1
fi

LINES=$(cat "${REF_FILE}" | wc -l | tr -d ' ')
GO_FILES=$(find . -name "*.go" | grep -v -e "vendor")

for FILE in ${GO_FILES}; do
  DIFFER=$(cat "${FILE}" | sed 's/2015/2014/g;s/2016/2014/g' | head "-${LINES}" | diff -q - "${REF_FILE}")

  if [[ ! -z "${DIFFER}" ]]; then
    echo "${FILE} does not have the correct copyright notice."
    exit 1
  fi
done
