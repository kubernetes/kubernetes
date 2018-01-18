#!/bin/bash

# Copyright 2017 The Kubernetes Authors.
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

# Running cmd $RETRIES times in case of failures.
function run-cmd-with-retries {
  RETRIES="${RETRIES:-3}"
  for attempt in $(seq 1 ${RETRIES}); do
    local ret_val=0
    exec 5>&1 # Duplicate &1 to &5 for use below.
    # We don't use 'local' to declare result as then ret_val always gets value 0.
    # We use tee to output to &5 (redirected to stdout) while also storing it in the variable.
    result=$("$@" 2>&1 | tee >(cat - >&5)) || ret_val="$?"
    if [[ "${ret_val:-0}" -ne "0" ]]; then
      if [[ $(echo "${result}" | grep -c "already exists") -gt 0 ]]; then
        if [[ "${attempt}" == 1 ]]; then
          echo -e "${color_red}Failed to $1 $2 $3 as the resource hasn't been deleted from a previous run.${color_norm}" >& 2
          exit 1
        fi
        echo -e "${color_yellow}Succeeded to $1 $2 $3 in the previous attempt, but status response wasn't received.${color_norm}"
        return 0
      fi
      echo -e "${color_yellow}Attempt $attempt failed to $1 $2 $3. Retrying.${color_norm}" >& 2
      sleep $(($attempt * 5))
    else
      echo -e "${color_green}Succeeded to $1 $2 $3.${color_norm}"
      return 0
    fi
  done
  echo -e "${color_red}Failed to $1 $2 $3.${color_norm}" >& 2
  exit 1
}
