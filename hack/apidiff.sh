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

# Install apidiff and make sure it's found.
export GOBIN="${KUBE_TEMP}"
PATH="${GOBIN}:${PATH}"
echo "Installing apidiff into ${GOBIN}."
go install golang.org/x/exp/cmd/apidiff@latest

# Build api-changelog tool
echo "Building api-changelog tool."
go install ./hack/api-changelog

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

# These are grep extended regular expressions matching known harmless incompatible changes
# in client-go. For example, Kubernetes API changes imply changing the API's client-go typed interfaces.
#
# - ./informers/resource/v1beta2.Interface.DeviceTaintRules: added
# - ./kubernetes/typed/resource/v1beta2.DeviceTaintRulesGetter.DeviceTaintRules: added
# ...
# - ./informers/autoscaling.Interface.V2beta1: removed
# - ./informers/autoscaling.Interface.V2beta2: removed
# ...
# - package k8s.io/client-go/applyconfigurations/autoscaling/v2beta1: removed
# - package k8s.io/client-go/applyconfigurations/autoscaling/v2beta2: removed
# ...
# - package k8s.io/client-go/kubernetes/typed/autoscaling/v2beta1: removed
# ...
# - package k8s.io/client-go/listers/autoscaling/v2beta2: removed
# ...
# - ./applyconfigurations/core/v1.PodCertificateProjectionApplyConfiguration: old is comparable, new is not
# - ./informers/certificates/v1alpha1.NewFilteredPodCertificateRequestInformer: removed
# - ./kubernetes/typed/certificates/v1alpha1.(*CertificatesV1alpha1Client).PodCertificateRequests: removed
# - ./kubernetes/typed/certificates/v1alpha1.PodCertificateRequestsGetter.PodCertificateRequests, method set of CertificatesV1alpha1Interface: removed
# - ./kubernetes/typed/certificates/v1alpha1/fake.(*FakeCertificatesV1alpha1).PodCertificateRequests: removed
#
# - ./kubernetes/typed/certificates/v1alpha1.PodCertificateRequestsGetter.PodCertificateRequests, method set of CertificatesV1alpha1Interface: removed
incompatible_filters=(
    -e '^- \./(informers|kubernetes/typed)/[a-z]+/v[a-z0-9]+(\.[a-zA-Z0-9]+)?\.[a-zA-Z0-9]+(, method set of .*)?: (added|removed|old is comparable, new is not)$'
    -e '^- \./kubernetes(\.Interface|/fake...Clientset.|...Clientset.)\.[A-Za-z]+V[a-z0-9]+(, method set of .*)?: (added|removed)$'
    -e '^- \./kubernetes/typed/[a-z]+/v[a-z0-9]+(/fake)?\.(..[a-zA-Z0-9]+.|[a-zA-Z0-9]+)\.[a-zA-Z0-9]+(, method set of .*)?: (added|removed|old is comparable, new is not)$'
    -e '^- \./informers/[a-z]+\.Interface\.V[a-z0-9]+(, method set of .*)?: (added|removed)$'
    -e '^- \./(listers|applyconfigurations)/[a-z]+/v1.*(, method set of .*)?: (added|removed|old is comparable, new is not)$'
    -e '^- package k8s\.io/client-go/(applyconfigurations|informers|kubernetes/typed|listers)/[a-z]+/v[a-z0-9]+(/fake)?: (added|removed)$'
)

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
    changes=$(apidiff -m "${before}" "${after}" 2>&1 | grep -v -e "^Ignoring internal package") || true
    incompatible=
    echo "## ${what}"
    if [ -z "$changes" ]; then
        echo "no changes"
    else
        # The output contains incompatible changes first, then compatible ones.
        # Both are optional. To find exactly the incompatible ones, we first
        # drop the compatible ones (if present) at the end, then look for the
        # incompatible changes. What's left is the header.
        #
        # The content of each section is unsorted. We fix this via sorting
        # the lines within each section because it makes the output more
        # predictable and is crucial for comparison of the incompatible
        # changes against the CHANGELOG.md (if there is any).
        sep=$(echo "$changes" | grep -n '^Compatible changes:$' | sed -e 's/:.*//') || true
        compatible=
        if [ -n "$sep" ]; then
            compatible=$(echo "$changes" | tail -n "+$((sep + 1))" | LC_ALL=C sort) || true
            changes=$(echo "$changes" | head -n "$((sep-1))") || true
        fi
        sep=$(echo "$changes" | grep -n '^Incompatible changes:$' | sed -e 's/:.*//') || true
        tolerated=
        if [ -n "$sep" ]; then
            # This is where we filter out certain known harmless changes.
            #
            # We can do that here in a generic script because the regular expressions for client-go
            # are unlikely to match incorrectly in a different component. If this ever changes,
            # then we can also store per-component filters in special file in
            # the component and load them from there.
            incompatible=$(echo "$changes" | tail -n "+$((sep + 1))" | LC_ALL=C sort) || true
            tolerated=$(echo "$incompatible" | grep -E "${incompatible_filters[@]}") || true
            incompatible=$(echo "$incompatible" | grep -v -E "${incompatible_filters[@]}") || true
            changes=$(echo "$changes" | head -n "$((sep-1))") || true
        fi
        # One of these strings must contain some change.
        echo "$changes"
        if [ -n "$incompatible" ]; then
            echo "Incompatible changes:"
            echo "${incompatible}"
        fi
        if [ -n "$tolerated" ]; then
            echo "Acceptable incompatible changes:"
            echo "${tolerated}"
        fi
        if [ -n "$compatible" ]; then
            echo "Compatible changes:"
            echo "${compatible}"
        fi

        echo
    fi
    if [ -n "$incompatible" ]; then
        # Does this directory have a changelog?
        # If yes, then maybe it already contains this incompatible change.
        changelog="${what}/${CHANGELOG}"
        if [ -f "${changelog}" ]; then
            # Use api-changelog tool to verify that incompatible changes are documented.
            # Exit codes: 0=success, 1=error, 2=verification failed
            set +e
            verify_output=$(api-changelog -verify -changelog="${changelog}" -changes="${incompatible}" 2>&1)
            verify_result=$?
            set -e
            if [ ${verify_result} -eq 0 ]; then
                # Documented => don't track it as a reason for failure.
                return 0
            elif [ ${verify_result} -eq 1 ]; then
                # Unexpected error from api-changelog tool
                echo "ERROR: api-changelog verification failed with unexpected error:" >&2
                echo "${verify_output}" >&2
                exit 1
            fi
            # verify_result == 2 means changes not found, continue to handle below.
            if ${update_changelog}; then
                # Use api-changelog tool to insert the changes into the changelog.
                if ${from_merge_commit}; then
                    # Example for a body:
                    #
                    # Merge pull request #137170 from pohly/dra-device-taints-beta
                    #
                    # DRA device taints: graduate to beta
                    #
                    # Parsing this is good enough for actual merge commits in Kubernetes 1.36.
                    # It's not meant to catch errors or unexpected body content.
                    body=$(git show --no-patch --format=%B "${merge_commit}")
                    # shellcheck disable=SC2207 # Here we intentionally split into words.
                    commit_summary=( $(echo "${body}" | head -n 1) )
                    pr=${commit_summary[3]}
                    pr=${pr#?}
                    title=$(echo "${body}" | tail -n 1)
                    description="See [PR #${pr}](https://github.com/kubernetes/kubernetes/pull/${pr})."
                    api-changelog -insert -changelog="${changelog}" -changes="${incompatible}" -title="${title}" -description="${description}"
                else
                    api-changelog -insert -changelog="${changelog}" -changes="${incompatible}"
                fi

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
    echo
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
