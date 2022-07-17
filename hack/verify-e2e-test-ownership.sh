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

# This script verifies the following e2e test ownership policies
# - tests MUST start with [sig-foo]
# - tests SHOULD NOT have multiple [sig-foo] tags
# TODO: these two can be dropped if KubeDescribe is gone from codebase
# - tests MUST NOT have [k8s.io] in test names
# - tests MUST NOT use KubeDescribe

set -o errexit
set -o nounset
set -o pipefail

# This will canonicalize the path
KUBE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd -P)
source "${KUBE_ROOT}/hack/lib/init.sh"

# Set REUSE_BUILD_OUTPUT=y to skip rebuilding dependencies if present
REUSE_BUILD_OUTPUT=${REUSE_BUILD_OUTPUT:-n}
# set VERBOSE_OUTPUT=y to output .jq files and shell commands
VERBOSE_OUTPUT=${VERBOSE_OUTPUT:-n}

if [[ ${VERBOSE_OUTPUT} =~ ^[yY]$ ]]; then
  set -x
fi

pushd "${KUBE_ROOT}" > /dev/null

# Setup a tmpdir to hold generated scripts and results
readonly tmpdir=$(mktemp -d -t verify-e2e-test-ownership.XXXX)
trap 'rm -rf ${tmpdir}' EXIT

# input
spec_summaries="${KUBE_ROOT}/_output/specsummaries.json"
# output
results_json="${tmpdir}/results.json"
summary_json="${tmpdir}/summary.json"
failures_json="${tmpdir}/failures.json"

# rebuild dependencies if necessary
function ensure_dependencies() {
  local -r ginkgo="${KUBE_ROOT}/_output/bin/ginkgo"
  local -r e2e_test="${KUBE_ROOT}/_output/bin/e2e.test"
  if ! { [ -f "${ginkgo}" ] && [[ "${REUSE_BUILD_OUTPUT}" =~ ^[yY]$ ]]; }; then
    make ginkgo
  fi
  if ! { [ -f "${e2e_test}" ] && [[ "${REUSE_BUILD_OUTPUT}" =~ ^[yY]$ ]]; }; then
    hack/make-rules/build.sh test/e2e/e2e.test
  fi
  if ! { [ -f "${spec_summaries}" ] && [[ "${REUSE_BUILD_OUTPUT}" =~ ^[yY]$ ]]; }; then
    "${ginkgo}" --dry-run=true "${e2e_test}" -- --spec-dump "${spec_summaries}" > /dev/null
  fi
}

# evaluate ginkgo spec summaries against e2e test ownership polices
# output to ${results_json}
function generate_results_json() {
  readonly results_jq=${tmpdir}/results.jq
  cat >"${results_jq}" <<EOS
  [.[] |  select( .LeafNodeType == "It") | . as { ContainerHierarchyTexts: \$text, ContainerHierarchyLocations: \$code, LeafNodeText: \$leafText,  LeafNodeLocation: \$leafCode} | {
      calls: ([ \$text | range(0;length) as \$i | {
        sig: ((\$text[\$i] | match("\\\[(sig-[^\\\]]+)\\\]") | .captures[0].string) // "unknown"),
        text: \$text[\$i],
        # unused, but if we ever wanted to have policies based on other tags...
        # tags: \$text[\$i] | [match("(\\\[[^\\\]]+\\\])"; "g").string],
        line: \$code[\$i] | "\(.FileName):\(.LineNumber)"
      }] + [{
        sig: ((\$leafText | match("\\\[(sig-[^\\\]]+)\\\]") | .captures[0].string) // "unknown"),
        text: \$leafText,
        # unused, but if we ever wanted to have policies based on other tags...
        # tags: \$leafText | [match("(\\\[[^\\\]]+\\\])"; "g").string],
        line: \$leafCode | "\(.FileName):\(.LineNumber)"
      }]),
    } | {
      owner: .calls[0].sig,
      calls: .calls,
      testname: .calls | map(.text) | join(" "),
      policies: [(
        .calls[0] |
          {
            fail: (.sig == "unknown"),
            level: "FAIL",
            category: "unowned_test",
            reason: "must start with [sig-foo]",
            found: .,
          }
        ), (
        .calls[1:] |
          (map(select(.sig != "unknown")) // [] | {
            fail: . | any,
            level: "WARN",
            category: "too_many_sigs",
            reason: "should not have multiple [sig-foo] tags",
            found: .,
          })
        )
      ]
  }]
EOS
  if [[ ${VERBOSE_OUTPUT} =~ ^[yY]$ ]]; then
    echo "about to  ${results_jq}..."
    cat -n "${results_jq}"
    echo
  fi
  <"${spec_summaries}" jq --slurp --from-file "${results_jq}" > "${results_json}"
}

# summarize e2e test policy results
# output to ${summary_json}
function generate_summary_json() {
  summary_jq=${tmpdir}/summary.jq
  cat >"${summary_jq}" <<EOS
  . as \$results |
  # for each policy category
  reduce \$results[0].policies[] as \$p ({}; . + {
    # add a convenience .policy field containing that policy's result
    (\$p.category): \$results | map(. + {policy: .policies[] | select(.category == \$p.category)}) | {
      level: \$p.level,
      reason: \$p.reason,
      passing: map(select(.policy.fail | not)) | length,
      failing: map(select(.policy.fail)) | length,
      testnames: map(select(.policy.fail) | .testname),
    }
  })
  # add a meta policy based on whether any policy failed
  + {
    all_policies: \$results | {
      level: "WARN",
      reason: "should pass all policies",
      passing: map(select(.policies | map(.fail) | any | not)) | length,
      failing: map(select(.policies | map(.fail) | any)) | length,
      testnames: map(select(.policies | map(.fail) | any) | .testname),
    }
  }
  # if a policy has no failing tests, change its log output to PASS
  | with_entries(.value += { log: (if (.value.failing == 0) then "PASS" else .value.level end) })
  # sort by policies with the most failing tests first
  | to_entries | sort_by(.value.failing) | reverse | from_entries
EOS
  if [[ ${VERBOSE_OUTPUT} =~ ^[yY]$ ]]; then
    echo "about to run ${results_jq}..."
    cat -n "${summary_jq}"
    echo
  fi
  <"${results_json}" jq --from-file "${summary_jq}" > "${summary_json}"
}

# filter e2e policy tests results to tests that failed, with the policies they failed
# output to ${failures_json}
function generate_failures_json() {
  local -r failures_jq="${tmpdir}/failures.jq"
  cat >"${failures_jq}" <<EOS
  .
  # for each test
  | map(
    # filter down to failing policies; trim category, .reason is more verbose
    .policies |= map(select(.fail) | del(.category))
    # trim the full callstack, .found will contain the relevant call
    | del(.calls)
  )
  # filter down to tests that have failed policies
  | map(select(.policies | map (.fail) | any))
EOS
  if [[ ${VERBOSE_OUTPUT} =~ ^[yY]$ ]]; then
    echo "about to run ${failures_jq}..."
    cat -n "${failures_jq}"
    echo
  fi
  <"${results_json}" jq --from-file "${failures_jq}" > "${failures_json}"
}

function output_results_and_exit_if_failed() {
  local -r total_tests=$(<"${spec_summaries}" wc -l | awk '{print $1}')

  # output results to console
  (
    echo "run at datetime: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "based on commit: $(git log -n1 --date=iso-strict --pretty='%h - %cd - %s')"
    echo
    <"${failures_json}" cat
    printf "%4s: e2e tests %-40s: %-4d\n" "INFO" "in total" "${total_tests}"
    <"${summary_json}" jq -r 'to_entries[].value |
      "printf \"%4s: ..failing %-40s: %-4d\\n\" \"\(.log)\" \"\(.reason)\" \"\(.failing)\""' | sh
  ) | tee "${tmpdir}/output.txt"
  # if we said "FAIL" in that output, we should fail
  if <"${tmpdir}/output.txt" grep -q "^FAIL"; then
    echo "FAIL"
    exit 1
  fi
}

ensure_dependencies
generate_results_json
generate_failures_json
generate_summary_json
output_results_and_exit_if_failed
echo "PASS"
