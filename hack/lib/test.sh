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

# A set of helpers for tests

readonly reset=$(tput sgr0)
readonly  bold=$(tput bold)
readonly black=$(tput setaf 0)
readonly   red=$(tput setaf 1)
readonly green=$(tput setaf 2)

kube::test::clear_all() {
  kubectl delete "${kube_flags[@]}" rc,pods --all
}

kube::test::get_object_assert() {
  local object=$1
  local request=$2
  local expected=$3

  res=$(kubectl get "${kube_flags[@]}" $object -o template -t "$request")

  if [[ "$res" =~ ^$expected$ ]]; then
      echo -n ${green}
      echo "Successful get $object $request: $res"
      echo -n ${reset}
      return 0
  else
      echo ${bold}${red}
      echo "FAIL!"
      echo "Get $object $request"
      echo "  Expected: $expected"
      echo "  Got:      $res"
      echo ${reset}${red}
      caller
      echo ${reset}
      return 1
  fi
}
