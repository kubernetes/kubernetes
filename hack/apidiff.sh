#!/usr/bin/env bash

# Copyright 2024 The Kubernetes Authors.
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

# This script checks the coding style for the Go language files using
# golangci-lint. Which checks are enabled depends on command line flags. The
# default is a minimal set of checks that all existing code passes without
# issues.

usage () {
  cat <<EOF >&2
Usage: $0 [-r <revision>] [package]"
   -r <revision>: only report issues in code added since that revision. Default is
      the common base of origin/master and HEAD.
   [package]: check specific package or directory instead of everything
EOF
  exit 1
}

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

base="$(git merge-base origin/master HEAD)"
while getopts "ar:sng:c:" o; do
    case "${o}" in
        r)
            base="${OPTARG}"
            if [ ! "$base" ]; then
                echo "ERROR: -c needs a non-empty parameter" >&2
                echo >&2
                usage
            fi
            ;;
        *)
            usage
            ;;
    esac
done

targets=("$@")

# Must be a something that git can resolve to a commit.
# "git rev-parse --verify" checks that and prints a detailed
# error.
base="$(git rev-parse --verify "$base")"

kube::golang::verify_go_version

# Needed to install and run apidiff, turned off by the setup scripts.
GO111MODULE=on

# Install apidiff and make sure it's found.
export GOBIN="${KUBE_OUTPUT_BINPATH}"
PATH="${GOBIN}:${PATH}"
echo "installing apidiff into ${GOBIN}"
(
    cd "${KUBE_ROOT}/hack/tools"
    go install golang.org/x/exp/cmd/apidiff
)

cd "${KUBE_ROOT}"

kube::util::ensure-temp-dir

run () {
    out="$1"
    mkdir -p "$out"
    if [[ "${#targets[@]}" -gt 0 ]]; then
        apidiff -m -w "${out}/all.out" "${targets[@]}"
    else
        apidiff -m -w "${out}/all.out" .
        for d in staging/src/k8s.io/*; do
            (
                cd "${d}"
                apidiff -m -w "${out}/$(basename "${d}").out" .
            )
        done
    fi
}

# First the current code.
run "${KUBE_TEMP}/after"

WORKTREE="${KUBE_TEMP}/worktree"

# Create a copy of the repo with the base checked out.
git worktree add -f -d "${WORKTREE}" "${base}"
# Clean up the copy on exit.
kube::util::trap_add "git worktree remove -f ${WORKTREE}" EXIT

# Base ready for apidiff.
(
    cd "${WORKTREE}"
    run "${KUBE_TEMP}/before"
)

# Now produce a report. All changes get reported because exporting some API
# unnecessarily might also be good to know, but the final exit code will only
# be non-zero if there are incompatible changes.
#
# The report is Markdown-formatted and can be copied into a PR comment verbatim.
res=0
echo
compare () {
    what="$1"
    before="$2"
    after="$3"
    changes=$(apidiff -m "${before}" "${after}" 2>&1 | grep -v -e "^Ignoring internal package") || true
    if [ -n "$changes" ]; then
        echo "## ${what}"
        echo "$changes"
        echo
    fi
    incompatible=$(apidiff -incompatible -m "${before}" "${after}" 2>&1) || true
    if [ -n "$incompatible" ]; then
        res=1
    fi
}

if [[ "${#targets[@]}" -gt 0 ]]; then
    compare "${targets[*]}" "${KUBE_TEMP}/before/all.out" "${KUBE_TEMP}/after/all.out"
else
    compare "k/k" "${KUBE_TEMP}/before/all.out" "${KUBE_TEMP}/after/all.out"
    for d in staging/src/k8s.io/*; do
        p=$(basename "${d}")
        before="${KUBE_TEMP}/before/${p}.out"
        after="${KUBE_TEMP}/after/${p}.out"
        if ! [ -e "${before}" ]; then
            echo "${d}: new package"
        else
            compare "${d}" "${before}" "${after}"
        fi
    done
fi

exit "$res"
