#!/bin/bash

# Copyright 2014 Google Inc. All rights reserved.
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

kube::util::sortable_date() {
  date "+%Y%m%d-%H%M%S"
}

kube::util::wait_for_url() {
  local url=$1
  local prefix=${2:-}
  local wait=${3:-0.2}
  local times=${4:-10}

  local i
  for i in $(seq 1 $times); do
    local out
    if out=$(curl -fs $url 2>/dev/null); then
      kube::log::status ${prefix}${out}
      return 0
    fi
    sleep $wait
  done
  kube::log::error "Timed out waiting for ${url}"
  return 1
}
