#!/bin/bash

# Copyright 2014 The Kubernetes Authors.
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

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/cluster/lib/util.sh"

# Excluded checks are always skipped.
EXCLUDED_CHECKS=(
  "verify-linkcheck.sh"  # runs in separate Jenkins job once per day due to high network usage
  "verify-govet.sh"      # it has a separate make vet target
  "verify-staging-client-go.sh" # TODO: enable the script after 1.5 code freeze
  )

function is-excluded {
  if [[ $1 -ef "$KUBE_ROOT/hack/verify-all.sh" ]]; then
    return
  fi
  for e in ${EXCLUDED_CHECKS[@]}; do
    if [[ $1 -ef "$KUBE_ROOT/hack/$e" ]]; then
      return
    fi
  done
  return 1
}

function run-cmd {
  if ${SILENT}; then
    "$@" &> /dev/null
  else
    "$@"
  fi
}

function run-checks {
  local -r pattern=$1
  local -r runner=$2

  for t in $(ls ${pattern})
  do
    if is-excluded "${t}" ; then
      echo "Skipping ${t}"
      continue
    fi
    echo -e "Verifying ${t}"
    local start=$(date +%s)
    run-cmd "${runner}" "${t}" && tr=$? || tr=$?
    local elapsed=$(($(date +%s) - ${start}))
    if [[ ${tr} -eq 0 ]]; then
      echo -e "${color_green}SUCCESS${color_norm}  ${t}\t${elapsed}s"
    else
      echo -e "${color_red}FAILED${color_norm}   ${t}\t${elapsed}s"
      ret=1
    fi
  done
}

while getopts ":v" opt; do
  case ${opt} in
    v)
      SILENT=false
      ;;
    \?)
      echo "Invalid flag: -${OPTARG}" >&2
      exit 1
      ;;
  esac
done

if ${SILENT} ; then
  echo "Running in silent mode, run with -v if you want to see script logs."
fi

ret=0
run-checks "${KUBE_ROOT}/hack/verify-*.sh" bash
run-checks "${KUBE_ROOT}/hack/verify-*.py" python
exit ${ret}

# ex: ts=2 sw=2 et filetype=sh
