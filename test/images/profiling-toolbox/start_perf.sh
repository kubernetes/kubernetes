#!/usr/bin/env bash

# Copyright 2018 The Kubernetes Authors.
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

# Start perf record command.
# Perf record perf_events inside the cgroup and output the result file to .data
# Usage: start_perf CGROUP
CGROUP=$1
perf record -F 99 -e cpu-clock -a -G $CGROUP -g -o perf.data 1> perf.out 2> perf.err &
PID=$!
echo $PID > /PID
