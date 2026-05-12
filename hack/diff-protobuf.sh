#!/usr/bin/env bash

# Copyright 2025 The Kubernetes Authors.
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

if [[ $# -ne 2 ]]; then
  echo "requires two arguments"
  echo ""
  echo "Examples:"
  echo "  hack/diff-protobuf.sh staging/src/k8s.io/api/testdata/v1.22.0/core.v1.CreateOptions{,.after_roundtrip}.pb"
  echo ""
  echo "  git difftool -x hack/diff-protobuf.sh --no-prompt <git diff args...>"
  echo "  git difftool -x hack/diff-protobuf.sh --no-prompt c0b7858946c253 4144c9294f7448 -- staging/src/k8s.io/api/testdata/HEAD/*.pb"
  exit 1
fi

lhs="${1}"
rhs="${2}"
if [[ ! -f "${lhs}" && ! -f "${rhs}" ]]; then
  echo "${lhs} and ${rhs} do not exist."
  exit 1
fi

# simple case
diffcmd="diff"
if [ -t 1 ]; then
  # if we're in a terminal, try to colorize
  if diff --help | grep color >/dev/null; then
    # diff supports --color
    diffcmd="diff --color=always"
  elif command -v colordiff &> /dev/null; then
    # alternative color diff command
    diffcmd="colordiff"
  fi
fi

echo "${BASE}"
if [[ "${lhs}" = *".pb" || "${rhs}" = *".pb" ]]; then
  if ! command -v protoc &> /dev/null
  then
      echo "protoc command not found"
      exit 1
  fi

  $diffcmd -u \
    <(tail -c +5 "${lhs}" | protoc --decode_raw) \
    <(tail -c +5 "${rhs}" | protoc --decode_raw) \
  | tail -n +3
else
  $diffcmd -u "${lhs}" "${rhs}" | tail -n +3
fi
echo ""
