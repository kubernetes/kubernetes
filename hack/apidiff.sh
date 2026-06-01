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

# This script analyzes API changes between specified revisions this repository.
# It uses the apidiff tool to detect differences, reports incompatible changes, and optionally
# builds downstream projects to assess the impact of those changes.
#
# Any directory with a CHANGELOG.md file must have incompatible changes documented
# in that file, otherwise the script fails with an error. If there are only
# compatible changes or all incompatible changes are documented, the script
# returns success.

usage () {
  cat <<EOF >&2
Usage: $0 [-r <revision>] [directory ...]"
   -t <revision>: Report changes in code up to and including this revision.
                  Default is the current working tree instead of a revision.
   -r <revision>: Report change in code added since this revision. Default is
                  the common base of origin/master and HEAD.
   -b <directory> Build all packages in that directory after replacing
                  Kubernetes dependencies with the current content of the
                  staging repo. May be given more than once. Must be an
                  absolute path.
                  WARNING: this will modify the go.mod in that directory.
   -u             Update changelog files if incompatible changes are found.
   -m             When enabled, -t must be used and must be given a merge commit
                  for a GitHub pull request. The diff is then calculated for
                  the merged branch. When updating the changelog, populates
                  the new changelog section with information about the pull request.
                  -r is ignored.
   [directory]:   Check one or more specific directory instead of everything.

EOF
  exit 1
}

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"
cd "${KUBE_ROOT}"

base=
target=
builds=()
update_changelog=false
from_merge_commit=false
while getopts "r:t:b:um" o; do
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
       b)
            if [ ! "${OPTARG}" ]; then
                echo "ERROR: -${o} needs a non-empty parameter" >&2
                echo >&2
                usage
            fi
            builds+=("${OPTARG}")
            ;;
       u)
            update_changelog=true
            ;;
       m)
            from_merge_commit=true
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND - 1))

if ${from_merge_commit}; then
    # Populating the changelog from merge commits is not a common
    # operation and thus skips detailed error handling.
    #
    # shellcheck disable=SC2207 # Here we intentionally split into words.
    parents=( $(git show --no-patch --format=%P "$target") )
    base=$(git merge-base "${parents[0]}" "${parents[1]}")
    merge_commit=${target}
    target=${parents[1]}
fi

# default from prow env if unset from args
# https://docs.prow.k8s.io/docs/jobs/#job-environment-variables
# TODO: handle batch PR testing

if [[ -z "${target:-}" && -n "${PULL_PULL_SHA:-}" ]]; then
    target="${PULL_PULL_SHA}"
fi
# target must be a something that git can resolve to a commit.
# "git rev-parse --verify" checks that and prints a detailed
# error.
if [[ -n "${target}" ]]; then
    target="$(git rev-parse --verify "${target}")"
fi

if [[ -z "${base}" && -n "${PULL_BASE_SHA:-}" && -n "${PULL_PULL_SHA:-}" ]]; then
    if ! base="$(git merge-base "${PULL_BASE_SHA}" "${PULL_PULL_SHA}")"; then
        echo >&2 "Failed to detect base revision correctly with prow environment variables."
        exit 1
    fi
elif [[ -z "${base}" ]]; then
    # origin is the default remote, but we encourage our contributors
    # to have both origin (their fork) and upstream, if upstream is present
    # then prefer upstream
    # if they have called it something else, there's no good way to be sure ...
    remote='origin'
    if git remote | grep -q 'upstream'; then
        remote='upstream'
    fi
    default_branch="$(git rev-parse --abbrev-ref "${remote}"/HEAD | cut -d/ -f2)"
    if ! base="$(git merge-base "${remote}/${default_branch}" "${target:-HEAD}")"; then
        echo >&2 "Could not determine default base revision. -r must be used explicitly."
        exit 1
    fi
fi
base="$(git rev-parse --verify "${base}")"

# Check specific directory or everything.
targets=("$@")
if [ ${#targets[@]} -eq 0 ]; then
    shopt -s globstar
    # Modules are discovered by looking for go.mod rather than asking go
    # to ensure that modules that aren't part of the workspace and/or are
    # not dependencies are checked too.
    # . and staging are listed explicitly here to avoid _output
    for module in ./go.mod ./staging/**/go.mod; do
        module="${module%/go.mod}"
        targets+=("$module")
    done
fi

# Give some information about what's happening. Failures from "git describe" are ignored
# silently, that's optional information.
describe () {
    local rev="$1"
    local descr
    echo -n "$rev"
    if descr=$(git describe --tags "${rev}" 2>/dev/null); then
        echo -n " (= ${descr})"
    fi
    echo
}
echo "Checking $(if [ -n "${target}" ]; then describe "${target}"; else echo "current working tree"; fi) for API changes since $(describe "${base}")."

kube::golang::setup_env
kube::util::ensure-temp-dir

# Install tools and make sure they are found.
export GOBIN="${KUBE_TEMP}"
PATH="${GOBIN}:${PATH}"
echo "Installing apidiff into ${GOBIN}."
go install golang.org/x/exp/cmd/apidiff@latest
echo "Installing apidiff-changelog into ${GOBIN}."
go install ./hack/apidiff-changelog

# tryBuild checks whether some other project builds with the staging repos
# of the current Kubernetes directory.
tryBuild () {
    local build="$1"

    # Replace all staging repos, whether the project uses them or not (playing it safe...).
    local repo
    for repo in $(cd staging/src; find k8s.io -name go.mod); do
        local path
        repo=$(dirname "${repo}")
        path="$(pwd)/staging/src/${repo}"
        (
            cd "$build"
            go mod edit -replace "${repo}"="${path}"
        )
    done

    # We only care about building. Breaking compilation of unit tests is also
    # annoying, but does not affect downstream consumers.
    (
        cd "$build"
        rm -rf vendor
        go mod tidy
        go build ./...
    )
}

# Build the flags for apidiff-changelog.
apidiff_flags=(-base="${base}")
if [ -n "${target}" ]; then
    apidiff_flags+=(-target="${target}")
fi
if ${update_changelog}; then
    apidiff_flags+=(-update-changelog)
fi
if ${from_merge_commit}; then
    apidiff_flags+=(-merge-commit="${merge_commit}")
fi

res=0
(set -x; apidiff-changelog "${apidiff_flags[@]}" "${targets[@]}") && apidiff_exit=0 || apidiff_exit=$?

# Fail if apidiff-changelog failed, unless the exit code indicates that all
# incompatible changes were documented.
if [ ${apidiff_exit} -ne 0 ] && [ ${apidiff_exit} -ne 3 ]; then
    res=1
fi

# Were any incompatible changes detected?
if [ "${apidiff_exit}" -gt 1 ] && [ ${#builds[@]} -gt 0 ]; then
    cat <<EOF

To help with assessing the real-world impact of an
API change, $0 will now try to build code in
${builds[@]}.
EOF

    if [[ "${builds[*]}" =~ controller-runtime ]]; then
cat <<EOF

controller-runtime is used because
- It tends to use advanced client-go functionality.
- Breaking it has additional impact on controller
  built on top of it.

This doesn't mean that an API change isn't allowed
if it breaks controller runtime, it just needs additional
scrutiny.

https://github.com/kubernetes-sigs/controller-runtime?tab=readme-ov-file#compatibility
explicitly states that a controller-runtime
release cannot be expected to work with a newer
release of the Kubernetes Go packages.
EOF
    fi

    for build in "${builds[@]}"; do
        echo
        echo "vvvvvvvvvvvvvvvv ${build} vvvvvvvvvvvvvvvvvv"
        if tryBuild "${build}"; then
            echo "${build} builds without errors."
        else
            cat <<EOF

WARNING: Building ${build} failed. This may or may not be because of the API changes!
EOF
        fi
        echo "^^^^^^^^^^^^^^^^ ${build} ^^^^^^^^^^^^^^^^^^"
    done
fi

exit "$res"
