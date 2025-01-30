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

# Short-circuit if verify-generated.sh has already been sourced.
[[ $(type -t kube::verify::generated::loaded) == function ]] && return 0

source "${KUBE_ROOT}/hack/lib/init.sh"

# This function verifies whether generated files are up-to-date. The first two
# parameters are messages that get printed to stderr when changes are found,
# the rest are the function or command and its parameters for generating files
# in the work tree.
#
# Example: kube::verify::generated "Mock files are out of date" "Please run 'hack/update-mocks.sh'" hack/update-mocks.sh
kube::verify::generated() {
  ( # a subshell prevents environment changes from leaking out of this function
    local failure_header=$1
    shift
    local failure_tail=$1
    shift

    kube::util::ensure_clean_working_dir

    # This sets up the environment, like GOCACHE, which keeps the worktree cleaner.
    kube::golang::setup_env

    _tmpdir="$(kube::realpath "$(mktemp -d -t "verify-generated-$(basename "$1").XXXXXX")")"
    git worktree add -f -q "${_tmpdir}" HEAD
    kube::util::trap_add "git worktree remove -f ${_tmpdir}" EXIT
    cd "${_tmpdir}"

    # Update generated files.
    "$@"

    # Test for diffs
    diffs=$(git status --porcelain | wc -l)
    if [[ ${diffs} -gt 0 ]]; then
      if [[ -n "${failure_header}" ]]; then
        echo "${failure_header}" >&2
      fi
      git status >&2
      git diff >&2
      if [[ -n "${failure_tail}" ]]; then
        echo "" >&2
        echo "${failure_tail}" >&2
      fi
      return 1
    fi
  )
}

# Marker function to indicate verify-generated.sh has been fully sourced.
kube::verify::generated::loaded() {
  return 0
}
