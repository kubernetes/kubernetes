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

set -o errexit
set -o nounset
set -o pipefail

# TODO: this list should be empty.
exceptions=("vendor/k8s.io/heapster/metrics/apis/metrics/v1alpha1")

failed=false
for i in $(find vendor/ -type d); do
  exceptionFound=false
  for e in ${exceptions[@]}; do
    if [[ "${e}" == "${i}" ]]; then
      echo "Skipping known violator $i"
      exceptionFound=true
      break
    fi
  done
  if ( "${exceptionFound}" == "true" ); then
    continue
  fi

  deps=$(go list -f '{{range .Deps}}{{.}}{{"\n"}}{{end}}' ./$i 2> /dev/null | grep -v "k8s.io/kubernetes/vendor/" | grep "k8s.io/kubernetes" || echo "")
    if [ -n "${deps}" ]; then
    echo "Package ${i} has a cyclic dependency on the main repository."
    failed=true
  fi
done

if [[ "${failed}" == "true" ]]; then
  exit 1
fi
