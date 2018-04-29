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

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..

cd "${KUBE_ROOT}"
result=0

find_files() {
  find . -not \( \
      \( \
        -wholename './output' \
        -o -wholename './_output' \
        -o -wholename './_gopath' \
        -o -wholename './release' \
        -o -wholename './target' \
        -o -wholename '*/third_party/*' \
        -o -wholename '*/vendor/*' \
        -o -wholename './staging/src/k8s.io/client-go/*vendor/*' \
      \) -prune \
    \) -name '*.go'
}

checkGlog() {
  while read file; do
    if grep -q "github.com/golang/glog" $file; then
      if grep "glog.Warning(\"" $file | grep %; then
        echo $file
	let result+=1
      fi
      if grep "glog.Info(\"" $file | grep %; then
        echo $file
	let result+=1
      fi
      if grep "glog.Error(\"" $file | grep %; then
        echo $file
	let result+=1
      fi
      if grep ").Info(\"" $file | grep % | grep glog.V\(; then
        echo $file
	let result+=1
      fi
    fi
  done
  if (( ${result}>=0 )); then
    echo
    echo "Use glog.*f when a format string is passed"
  fi
  return ${result}
}

find_files | checkGlog
