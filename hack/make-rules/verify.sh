#!/usr/bin/env bash

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

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/../..
source "${KUBE_ROOT}/hack/lib/util.sh"

# If KUBE_JUNIT_REPORT_DIR is unset, and ARTIFACTS is set, then have them match.
if [[ -z "${KUBE_JUNIT_REPORT_DIR:-}" && -n "${ARTIFACTS:-}" ]]; then
    export KUBE_JUNIT_REPORT_DIR="${ARTIFACTS}"
fi

# include shell2junit library
source "${KUBE_ROOT}/third_party/forked/shell2junit/sh2ju.sh"

# Excluded check patterns are always skipped.
EXCLUDED_PATTERNS=(
  "verify-all.sh"                # this script calls the make rule and would cause a loop
  "verify-linkcheck.sh"          # runs in separate Jenkins job once per day due to high network usage
  "verify-*-dockerized.sh"       # Don't run any scripts that intended to be run dockerized
  )

# Exclude generated-files-remake in certain cases, if they're running in a separate job.
if [[ ${EXCLUDE_FILES_REMAKE:-} =~ ^[yY]$ ]]; then
  EXCLUDED_PATTERNS+=(
    "verify-generated-files-remake.sh" # run in a separate job
    )
fi

# Excluded checks for openshift/kubernetes fork that are always skipped.
EXCLUDED_PATTERNS+=(
  "verify-boilerplate.sh"            # Carries do not require boilerplate
  "verify-bazel.sh"                  # Bazel is not used downstream
  "verify-no-vendor-cycles.sh"       # Incompatible with the way many carries are specified
  "verify-publishing-bot.py"         # Verifies the upstream rules, which are not maintained in o/k
  "verify-staging-meta-files.sh"     # Staging meta files are not maintained downstream
)

# Skipped checks for openshift/kubernetes fork that need to be fixed.
#
# Where a check is excluded due to 'inconsistent behavior between
# local and ci execution', the fix will require finding a way to
# compare current and generated results without 'cp -a' since this
# command does not execute without error in downstream ci.
EXCLUDED_PATTERNS+=(
  "verify-codegen.sh"                # TODO(marun) Fix inconsistent behavior between local and ci execution
  "verify-generated-files-remake.sh" # TODO(marun) Is it worth fixing this check?
  "verify-generated-protobuf.sh"     # TODO(marun) Fix inconsistent behavior between local and ci execution
  "verify-golint.sh"                 # TODO(marun) Cleanup carried code
  "verify-hack-tools.sh"             # TODO(marun) Fix inconsistent behavior between local and ci execution
  "verify-openapi-spec.sh"           # TODO(marun) Fix inconsistent behavior between local and ci execution
  "verify-spelling.sh"               # TODO(marun) Need to ensure installation of misspell command
  "verify-staticcheck.sh"            # TODO(marun) Fix inconsistent behavior between local and ci execution
  "verify-vendor-licenses.sh"        # TODO(marun) Fix inconsistent behavior between local and ci execution
)

# Exclude typecheck in certain cases, if they're running in a separate job.
if [[ ${EXCLUDE_TYPECHECK:-} =~ ^[yY]$ ]]; then
  EXCLUDED_PATTERNS+=(
    "verify-typecheck.sh"              # runs in separate typecheck job
    "verify-typecheck-providerless.sh" # runs in separate typecheck job
    "verify-typecheck-dockerless.sh" # runs in separate typecheck job
    )
fi

# Exclude dependency checks in certain cases, if they're running in a separate job.
# From @cblecker: We can't change the variable name here, unless we update it throughout
#                 test-infra (and we would need to pick it backwards).
if [[ ${EXCLUDE_GODEP:-} =~ ^[yY]$ ]]; then
  EXCLUDED_PATTERNS+=(
    "verify-external-dependencies-version.sh" # runs in separate dependencies job
    "verify-vendor.sh"                        # runs in separate dependencies job
    "verify-vendor-licenses.sh"               # runs in separate dependencies job
    )
fi

# Exclude readonly package check in certain cases, aka, in periodic jobs we don't care and a readonly package won't be touched
if [[ ${EXCLUDE_READONLY_PACKAGE:-} =~ ^[yY]$ ]]; then
  EXCLUDED_PATTERNS+=(
    "verify-readonly-packages.sh"  # skip in CI, if env is set
    )
fi

# Only run whitelisted fast checks in quick mode.
# These run in <10s each on enisoc's workstation, assuming that
# `make` had already been run.
QUICK_PATTERNS+=(
  "verify-api-groups.sh"
  "verify-bazel.sh"
  "verify-boilerplate.sh"
  "verify-external-dependencies-version.sh"
  "verify-vendor-licenses.sh"
  "verify-gofmt.sh"
  "verify-imports.sh"
  "verify-pkg-names.sh"
  "verify-readonly-packages.sh"
  "verify-spelling.sh"
  "verify-staging-client-go.sh"
  "verify-staging-meta-files.sh"
  "verify-test-featuregates.sh"
  "verify-test-images.sh"
)

while IFS='' read -r line; do EXCLUDED_CHECKS+=("$line"); done < <(ls "${EXCLUDED_PATTERNS[@]/#/${KUBE_ROOT}\/hack\/}" 2>/dev/null || true)
while IFS='' read -r line; do QUICK_CHECKS+=("$line"); done < <(ls "${QUICK_PATTERNS[@]/#/${KUBE_ROOT}\/hack\/}" 2>/dev/null || true)
TARGET_LIST=()
IFS=" " read -r -a TARGET_LIST <<< "${WHAT:-}"

function is-excluded {
  for e in "${EXCLUDED_CHECKS[@]}"; do
    if [[ $1 -ef "${e}" ]]; then
      return
    fi
  done
  return 1
}

function is-quick {
  for e in "${QUICK_CHECKS[@]}"; do
    if [[ $1 -ef "${e}" ]]; then
      return
    fi
  done
  return 1
}

function is-explicitly-chosen {
  local name="${1#verify-}"
  name="${name%.*}"
  index=0
  for e in "${TARGET_LIST[@]}"; do
    if [[ "${e}" == "${name}" ]]; then
      TARGET_LIST[${index}]=""
      return
    fi
    index=$((index + 1))
  done
  return 1
}

function run-cmd {
  local filename="${2##*/verify-}"
  local testname="${filename%%.*}"
  local output="${KUBE_JUNIT_REPORT_DIR:-/tmp/junit-results}"
  local tr

  if ${SILENT}; then
    juLog -output="${output}" -class="verify" -name="${testname}" "$@" &> /dev/null
    tr=$?
  else
    juLog -output="${output}" -class="verify" -name="${testname}" "$@"
    tr=$?
  fi
  return ${tr}
}

# Collect Failed tests in this Array , initialize it to nil
FAILED_TESTS=()

function print-failed-tests {
  echo -e "========================"
  echo -e "${color_red:?}FAILED TESTS${color_norm:?}"
  echo -e "========================"
  for t in "${FAILED_TESTS[@]}"; do
      echo -e "${color_red}${t}${color_norm}"
  done
}

function run-checks {
  local -r pattern=$1
  local -r runner=$2

  local t
  for t in ${pattern}
  do
    local check_name
    check_name="$(basename "${t}")"
    if [[ -n ${WHAT:-} ]]; then
      if ! is-explicitly-chosen "${check_name}"; then
        continue
      fi
    else
      if is-excluded "${t}" ; then
        echo "Skipping ${check_name}"
        continue
      fi
      if ${QUICK} && ! is-quick "${t}" ; then
        echo "Skipping ${check_name} in quick mode"
        continue
      fi
    fi
    echo -e "Verifying ${check_name}"
    local start
    start=$(date +%s)
    run-cmd "${runner}" "${t}" && tr=$? || tr=$?
    local elapsed=$(($(date +%s) - start))
    if [[ ${tr} -eq 0 ]]; then
      echo -e "${color_green:?}SUCCESS${color_norm}  ${check_name}\t${elapsed}s"
    else
      echo -e "${color_red}FAILED${color_norm}   ${check_name}\t${elapsed}s"
      ret=1
      FAILED_TESTS+=("${t}")
    fi
  done
}

# Check invalid targets specified in "WHAT" and mark them as failure cases
function missing-target-checks {
  # In case WHAT is not specified
  [[ ${#TARGET_LIST[@]} -eq 0 ]] && return

  for v in "${TARGET_LIST[@]}"
  do
    [[ -z "${v}" ]] && continue

    FAILED_TESTS+=("${v}")
    ret=1
  done
}

SILENT=${SILENT:-false}
QUICK=${QUICK:-false}

if ${SILENT} ; then
  echo "Running in silent mode, run with SILENT=false if you want to see script logs."
fi

if ${QUICK} ; then
  echo "Running in quick mode (QUICK=true). Only fast checks will run."
fi

ret=0
run-checks "${KUBE_ROOT}/hack/verify-*.sh" bash
run-checks "${KUBE_ROOT}/hack/verify-*.py" python
missing-target-checks

if [[ ${ret} -eq 1 ]]; then
    print-failed-tests
fi
exit ${ret}

# ex: ts=2 sw=2 et filetype=sh
