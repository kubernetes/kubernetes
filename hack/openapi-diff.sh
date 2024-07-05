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

usage () {
  cat <<EOF >&2
Usage: $0 [-t <revision>] [-r <revision>]"
   -t <revision>: Report changes in code up to and including this revision.
                  Default is the current working tree instead of a revision.
   -r <revision>: Report change in code added since this revision. Default is
                  the common base of origin/master and HEAD.
EOF
  exit 1
}

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

base=
target=
while getopts "r:t:" o; do
    case "${o}" in
        r)
            base="${OPTARG}"
            if [ ! "$base" ]; then
                echo "ERROR: -${o} needs a non-empty parameter" >&2
                echo >&2
                usage
            fi
            ;;
       t)
            target="${OPTARG}"
            if [ ! "$target" ]; then
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

# Must be a something that git can resolve to a commit.
# "git rev-parse --verify" checks that and prints a detailed
# error.
target="$(git rev-parse --verify "${target:-HEAD}")"

# Determine defaults.
if [ -z "${base}" ]; then
    if ! base="$(git merge-base origin/master "${target}")"; then
        echo >&2 "Could not determine default base revision. -r must be used explicitly."
        exit 1
    fi
fi
base="$(git rev-parse --verify "${base}")"

kube::golang::setup_env
kube::util::ensure-temp-dir

# Install oasdiff and make sure it's found.
export GOBIN="${KUBE_TEMP}"
PATH="${GOBIN}:${PATH}"
echo "Installing oasdiff into ${GOBIN}."
go install github.com/tufin/oasdiff@latest

cd "${KUBE_ROOT}"

# Create a copy of the repo with the specific revision checked out.
readonly worktreebefore="${KUBE_TEMP}/before"
git worktree add -f -d "${worktreebefore}" "${base}"
# Clean up the copy on exit.
kube::util::trap_add "git worktree remove -f ${worktreebefore}" EXIT

# Create a copy of the repo with the specific revision checked out.
readonly worktreeafter="${KUBE_TEMP}/after"
git worktree add -f -d "${worktreeafter}" "${target}"
# Clean up the copy on exit.
kube::util::trap_add "git worktree remove -f ${worktreeafter}" EXIT

readonly apipath="api/openapi-spec/v3"
res=0
for file in "${worktreebefore}/${apipath}/"*.json; do
    filename=$(basename "${file}")
    echo "Checking file ${file}"
    if [ -f "${worktreeafter}/${apipath}/${filename}" ]; then
        oasdiff breaking "${file}" "${worktreeafter}/${apipath}/${filename}" || res=$?
    else
        echo "${filename} was removed"
    fi
done
exit "$res"


