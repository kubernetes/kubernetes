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

set -e
set -x

# Check usage.
if [ $# == 0 ]; then
  echo "USAGE: run.sh <hosts to run tests on>"
  exit 1
fi

# Don't run on trivial changes.
if ! git diff --name-only origin/master | grep -c -E "*.go|*.sh" &> /dev/null; then
  echo "This PR does not touch files that require integration testing. Skipping integration tests."
  exit 0
fi

# Build the runner.
go build github.com/google/cadvisor/integration/runner

# Run it.
HOSTS=$@
./runner --logtostderr $HOSTS
