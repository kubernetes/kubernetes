#!/usr/bin/env bash
# Copyright 2019 The Kubernetes Authors.
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

set -o errexit
set -o nounset
set -o pipefail

E2ELOG=${1}
errors_alpha=()
errors_beta=()

# e.g: I0531 22:41:04.314617      18 round_trippers.go:416] GET https://10.96.0.1:443/api/v1/nodes
for line in $(egrep "(GET|POST|PUT|DELETE|HEAD) http" ${E2ELOG}| awk -F "] " '{print $2}'| sort| uniq| grep "alpha")
do
    errors_alpha+=( "${line}" )
done

for line in $(egrep "(GET|POST|PUT|DELETE|HEAD) http" ${E2ELOG}| awk -F "] " '{print $2}'| sort| uniq| grep "beta")
do
    errors_beta+=( "${line}" )
done

if [ ${#errors_beta[@]} -ne 0 ]; then
  {
    echo "Errors of Aplha APIs:"
    for err in "${errors_alpha[@]}"; do
      echo "$err"
    done
    echo "Errors of Beta APIs:"
    for err in "${errors_beta[@]}"; do
      echo "$err"
    done
    echo
    echo 'The above Alpha/Beta APIs are called in conformance tests which should use stable APIs only'
    echo
  } >&2
  exit 1
fi

