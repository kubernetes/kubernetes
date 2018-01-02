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

# Check formatting on non Godep'd code.
GOFMT_PATHS=$(find . -not -wholename "*.git*" -not -wholename "*Godeps*" -not -wholename "*vendor*" -not -name "." -type d)

# Find any files with gofmt problems
BAD_FILES=$(gofmt -s -l $GOFMT_PATHS)

if [ -n "$BAD_FILES" ]; then
  echo "The following files are not properly formatted:"
  echo $BAD_FILES
  exit 1
fi
