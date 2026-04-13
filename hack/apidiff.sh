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

CHANGELOG="CHANGELOG.md"

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
   -u             Update ${CHANGELOG} files if incompatible changes are found.
   [directory]:   Check one or more specific directory instead of everything.

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
builds=()
update_changelog=false
while getopts "r:t:b:u" o; do
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
        *)
            usage
            ;;
    esac
done
shift $((OPTIND - 1))

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
#
# shellcheck disable=SC2317 # "Command appears to be unreachable" - gets called indirectly.
run () {
    out="$1"
    mkdir -p "$out"
    for d in "${targets[@]}"; do
        if ! [ -d "${d}" ]; then
            echo "module ${d} does not exist, skipping ..."
            continue
        fi
        # cd to the path for modules that are intree but not part of the go workspace
        # per example staging/src/k8s.io/code-generator/examples
        (
            cd "${d}"
            apidiff -m -w "${out}/$(output_name "${d}")" .
        ) &
    done
    wait
}

# inWorktree checks out a specific revision, then invokes the given
# command there.
#
# shellcheck disable=SC2317 # "Command appears to be unreachable" - gets called indirectly.
inWorktree () {
    local worktree="$1"
    shift
    local rev="$1"
    shift

    # Create a copy of the repo with the specific revision checked out.
    # Might already have been done before.
    if ! [ -d "${worktree}" ]; then
        git worktree add -f -d "${worktree}" "${rev}"
        # Clean up the copy on exit.
        kube::util::trap_add "git worktree remove -f ${worktree}" EXIT
    fi

    # Ready for apidiff.
    (
        cd "${worktree}"
        "$@"
    )
}

# inTarget runs the given command in the target revision of Kubernetes,
# checking it out in a work tree if necessary.
inTarget () {
    if [ -z "${target}" ]; then
        "$@"
    else
        inWorktree "${KUBE_TEMP}/target" "${target}" "$@"
    fi
}

# Dump old and new api state.
inTarget run "${KUBE_TEMP}/after"
inWorktree "${KUBE_TEMP}/base" "${base}" run "${KUBE_TEMP}/before"

# Now produce a report. All changes get reported because exporting some API
# unnecessarily might also be good to know, but the final exit code will only
# be non-zero if there are incompatible changes.
#
# The report is Markdown-formatted and can be copied into a PR comment verbatim.
failures=()
can_update_changelog=false
echo
compare () {
    what="$1"
    before="$2"
    after="$3"
    if [ ! -f "${before}" ] || [ ! -f "${after}" ]; then
        echo "can not compare changes, module didn't exist before or after"
        return
    fi
    # Output order is non-deterministic. Fix that by sorting.
    # Here its mostly cosmetic, but for checking for incompatible changes
    # in a changelog it really matters.
    changes=$(apidiff -m "${before}" "${after}" 2>&1 | grep -v -e "^Ignoring internal package" | LC=C sort) || true
    echo "## ${what}"
    if [ -z "$changes" ]; then
        echo "no changes"
    else
        echo "$changes"
        echo
    fi
    incompatible=$(apidiff -incompatible -m "${before}" "${after}" 2>&1 | grep -v -e "^Ignoring internal package" | LC=C sort) || true
    if [ -n "$incompatible" ]; then
        # Does this directory have a changelog?
        # If yes, then maybe it already contains this incompatible change.
        changelog="${what}/${CHANGELOG}"
        if [ -f "${changelog}" ]; then
            # This checks that the incompatible change is in a verbatim text blog of the changelog.
            # Any surrounding text (section header, human-readable explanation) must be checked
            # by reviewers of the new changelog entry.
            if [[ $(cat "${changelog}") = *"\`\`\`
${incompatible}
\`\`\`"* ]]; then
                # Documented => don't track it as a reason for failure.
                return 0
            fi
            if ${update_changelog}; then
                # Add a new section before the earliest one or, if there is none, at the end.
                # Add an empty line before or after the new text, depending on where we insert.
                # This way the most recent information is directly visible.
                line=$(grep -m 1 -n '^#' "${changelog}") || true
                if [[ -z "${line}" ]]; then
                    line=$(( $(wc -l <"${changelog}")  + 1 ))
                    before="
"
                    after=""
                else
                    line=${line/:*/}
                    before=""
                    after="
"
                fi
                head -n $((line - 1)) "${changelog}" >"${changelog}.tmp"
                cat >>"${changelog}.tmp" <<EOF
${before}### Replace with a short title

Replace this text with a short summary of the change
and how users of the package can deal with this breaking
change. If users are not expected to be affected, then
instead explain why.

\`\`\`
${incompatible}
\`\`\`${after}
EOF
                tail -n "+${line}" "${changelog}" >>"${changelog}.tmp"
                mv "${changelog}.tmp" "${changelog}"

                # Because we have updated the changelog as requested, we don't
                # need to describe how to do that nor treat it as a failure.
                return 0
            fi
            can_update_changelog=true
        fi
        failures+=("${what}")
    fi
}

for d in "${targets[@]}"; do
    compare "${d}" "${KUBE_TEMP}/before/$(output_name "${d}")" "${KUBE_TEMP}/after/$(output_name "${d}")"
done

# tryBuild checks whether some other project builds with the staging repos
# of the current Kubernetes directory.
#
# shellcheck disable=SC2317 # "Command appears to be unreachable" - gets called indirectly.
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

res=0
if [ ${#failures[@]} -gt 0 ]; then
    res=1
    echo "Detected incompatible changes on modules:"
    printf '%s\n' "${failures[@]}"
    cat <<EOF

Some notes about API differences:

Changes in internal packages are usually okay.
However, remember that custom schedulers
and scheduler plugins depend on pkg/scheduler/framework.

API changes in staging repos are more critical.
Try to avoid them as much as possible.
But sometimes changing an API is the lesser evil
and/or the impact on downstream consumers is low.
Use common sense and code searches.
EOF

    if [ ${#builds[@]} -gt 0 ]; then

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
            if inTarget tryBuild "${build}"; then
                echo "${build} builds without errors."
            else
                cat <<EOF

WARNING: Building ${build} failed. This may or may not be because of the API changes!
EOF
            fi
            echo "^^^^^^^^^^^^^^^^ ${build} ^^^^^^^^^^^^^^^^^^"
        done
    fi

    if ${can_update_changelog}; then
        cat <<EOF

Run the following command to add add the incompatible changes to
the ${CHANGELOG} file(s), edit the modified file(s) to
replace the template text in the new section at the top
with and explanation of the changes, then include the result
in the pull request for review:

    hack/apidiff.sh -u -r ${base} ${target:+-t ${target} }${targets[*]}
EOF
    fi
fi

exit "$res"
