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
source "${KUBE_ROOT}/hack/lib/util.sh"

# Excluded check patterns are always skipped.
EXCLUDED_PATTERNS=(
  "verify-all.sh"                # this script calls the make rule and would cause a loop
  "verify-linkcheck.sh"          # runs in separate Jenkins job once per day due to high network usage
  "verify-test-owners.sh"        # TODO(rmmh): figure out how to avoid endless conflicts
  "verify-*-dockerized.sh"       # Don't run any scripts that intended to be run dockerized
  )

# Only run whitelisted fast checks in quick mode.
# These run in <10s each on enisoc's workstation, assuming that
# `make` and `hack/godep-restore.sh` had already been run.
QUICK_PATTERNS+=(
  "verify-api-groups.sh"
  "verify-bazel.sh"
  "verify-boilerplate.sh"
  "verify-generated-files-remake"
  "verify-godep-licenses.sh"
  "verify-gofmt.sh"
  "verify-imports.sh"
  "verify-pkg-names.sh"
  "verify-readonly-packages.sh"
  "verify-staging-client-go.sh"
  "verify-test-images.sh"
  "verify-test-owners.sh"
)

EXCLUDED_CHECKS=$(ls ${EXCLUDED_PATTERNS[@]/#/${KUBE_ROOT}\/hack\/} 2>/dev/null || true)
QUICK_CHECKS=$(ls ${QUICK_PATTERNS[@]/#/${KUBE_ROOT}\/hack\/} 2>/dev/null || true)

function is-excluded {
  for e in ${EXCLUDED_CHECKS[@]}; do
    if [[ $1 -ef "$e" ]]; then
      return
    fi
  done
  return 1
}

function is-quick {
  for e in ${QUICK_CHECKS[@]}; do
    if [[ $1 -ef "$e" ]]; then
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

# Collect Failed tests in this Array , initalize it to nil
FAILED_TESTS=()

function print-failed-tests {
  echo -e "========================"
  echo -e "${color_red}FAILED TESTS${color_norm}"
  echo -e "========================"
  for t in ${FAILED_TESTS[@]}; do
      echo -e "${color_red}${t}${color_norm}"
  done
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
    if ${QUICK} && ! is-quick "${t}" ; then
      echo "Skipping ${t} in quick mode"
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
      FAILED_TESTS+=(${t})
    fi
  done
}

SILENT=true
QUICK=false

while getopts ":vQ" opt; do
  case ${opt} in
    v)
      SILENT=false
      ;;
    Q)
      QUICK=true
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

if ${QUICK} ; then
  echo "Running in quick mode (-Q flag). Only fast checks will run."
fi

ret=0
run-checks "${KUBE_ROOT}/hack/verify-*.sh" bash
run-checks "${KUBE_ROOT}/hack/verify-*.py" python

if [[ ${ret} -eq 1 ]]; then
    print-failed-tests 
fi
exit ${ret}

# ex: ts=2 sw=2 et filetype=sh
