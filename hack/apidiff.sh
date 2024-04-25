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
Usage: $0 [-r <revision>] [directory]"
   -r <revision>: only report issues in code added since that revision. Default is
      the common base of origin/master and HEAD.
   [package]: check directory instead of everything
EOF
  exit 1
}

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

base=
while getopts "r:" o; do
    case "${o}" in
        r)
            base="${OPTARG}"
            if [ ! "$base" ]; then
                echo "ERROR: -${o} needs a non-empty parameter" >&2
                echo >&2
                usage
            fi
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND - 1))

# Check specific directory or everything.
targets=("$@")
if [ ${#targets[@]} -eq 0 ]; then
    # This lists all entries in the go.work file as absolute directory paths.
    kube::util::read-array targets < <(
        go list -f '{{.Dir}}' -m | while read -r d; do
            # We need relative paths because we will invoke apidiff in
            # different work trees.
            d="$(realpath -s --relative-to="$(pwd)" "${d}")"
            if [ "${d}" != "." ]; then
                # sub-directories have to have a leading dot.
                d="./${d}"
            fi
            echo "${d}"
        done
    )
fi

# Default if -r was not given.
if [ -z "${base}" ]; then
    if ! base="$(git merge-base origin/master HEAD)"; then
        echo >&2 "Could not determine default base revision. -r must be used explicitly."
        exit 1
    fi
fi

# Must be a something that git can resolve to a commit.
# "git rev-parse --verify" checks that and prints a detailed
# error.
base="$(git rev-parse --verify "$base")"

# Give some information about what's happening. Failures from "git describe" are ignored
# silently, that's optional information.
echo "Checking for API changes since ${base}$(if descr=$(git describe --tags "${base}" 2>/dev/null); then echo " = ${descr}"; fi)."

kube::golang::setup_env
kube::util::ensure-temp-dir

# Install apidiff and make sure it's found.
export GOBIN="${KUBE_TEMP}"
PATH="${GOBIN}:${PATH}"
echo "Installing apidiff into ${GOBIN}."
go install golang.org/x/exp/cmd/apidiff@latest

cd "${KUBE_ROOT}"

# output_name targets a target directory and prints the base name of
# an output file for that target.
output_name () {
    what="$1"

    echo "${what}" | sed -e 's/[^a-zA-Z0-9_-]/_/g' -e 's/$/.out/'
}

# run invokes apidiff once per target and stores the output
# file(s) in the given directory.
run () {
    out="$1"
    mkdir -p "$out"
    for d in "${targets[@]}"; do
        apidiff -m -w "${out}/$(output_name "${d}")" "${d}"
    done
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
    echo "## ${what}"
    if [ -z "$changes" ]; then
        echo "no changes"
    else
        echo "$changes"
        echo
    fi
    incompatible=$(apidiff -incompatible -m "${before}" "${after}" 2>&1) || true
    if [ -n "$incompatible" ]; then
        res=1
    fi
}

for d in "${targets[@]}"; do
    compare "${d}" "${KUBE_TEMP}/before/$(output_name "${d}")" "${KUBE_TEMP}/after/$(output_name "${d}")"
done

exit "$res"
