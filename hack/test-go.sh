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


set -e

source $(dirname $0)/config-go.sh

# Go to the top of the tree.
cd "${KUBE_REPO_ROOT}"

# Check for `go` binary and set ${GOPATH}.
kube::setup_go_environment


find_test_dirs() {
  cd src/${KUBE_GO_PACKAGE}
  find . -not \( \
      \( \
        -wholename './output' \
        -o -wholename './_output' \
        -o -wholename './release' \
        -o -wholename './target' \
        -o -wholename '*/third_party/*' \
        -o -wholename '*/Godeps/*' \
      \) -prune \
    \) -name '*_test.go' -print0 | xargs -0n1 dirname | sort -u | xargs -n1 printf "${KUBE_GO_PACKAGE}/%s\n"
}

# -covermode=atomic becomes default with -race in Go >=1.3
KUBE_COVER=${KUBE_COVER:--cover -covermode=atomic}
KUBE_TIMEOUT=${KUBE_TIMEOUT:--timeout 30s}

cd "${KUBE_TARGET}"

while getopts "i:" opt ; do
  case $opt in
    i)
      iterations=$OPTARG
      ;;
    ?)
      echo "Invalid argument -$OPTARG"
      exit 1
      ;;
    :)
      echo "Option -$OPTARG <value>"
      exit 1
      ;;
  esac
done
shift $((OPTIND - 1))

if [[ -n "${iterations}" ]]; then
  echo "Running ${iterations} times"
  if [[ -n "$1" ]]; then
    pkg=$KUBE_GO_PACKAGE/$1
  fi
  rm -f *.test
  # build a test binary
  echo "${pkg}"
  go test -c -race ${KUBE_TIMEOUT} "${pkg}"
  # keep going, even if there are failures
  pass=0
  count=0
  for i in $(seq 1 ${iterations}); do
    for test_binary in *.test; do
      if "./${test_binary}"; then
        ((pass++))
      fi
      ((count++))
    done
  done 2>&1
  echo "${pass}" / "${count}" passing
  if [[ ${pass} != ${count} ]]; then
    exit 1
  else
    exit 0
  fi
fi

if [[ -n "$1" ]]; then
  go test ${GOFLAGS} \
      -race \
      ${KUBE_TIMEOUT} \
      ${KUBE_COVER} -coverprofile=tmp.out \
      "${KUBE_GO_PACKAGE}/$1" "${@:2}"
  exit 0
fi

find_test_dirs | xargs go test ${GOFLAGS:-} \
    -race \
    -timeout 30s \
    ${KUBE_COVER} \
    "${@:2}"
