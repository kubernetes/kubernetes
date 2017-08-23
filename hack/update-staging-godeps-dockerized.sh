#!/bin/bash

# Copyright 2017 The Kubernetes Authors.
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

# updates the godeps.json file in the staging folders to allow clean vendoring
# based on kubernetes levels.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

# Sets default values
DRY_RUN=false
FAIL_ON_DIFF=false
GODEP_OPTS=""

# Controls verbosity of the script output and logging.
KUBE_VERBOSE="${KUBE_VERBOSE:-5}"
if (( ${KUBE_VERBOSE} >= 6 )); then
  GODEP_OPTS+=("-v")
fi

while getopts ":df" opt; do
  case $opt in
    d) # do not godep-restore into a temporary directory, but use the existing GOPATH
      DRY_RUN=true
      ;;
    f) # fail if something in the Godeps.json files changed
      FAIL_ON_DIFF=true
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

kube::util::ensure_dockerized
kube::golang::setup_env
kube::util::ensure_single_dir_gopath
kube::util::ensure_no_staging_repos_in_gopath
kube::util::ensure_godep_version v79

kube::log::status "Checking whether godeps are restored"
if ! kube::util::godep_restored 2>&1 | sed 's/^/  /'; then
  ${KUBE_ROOT}/hack/godep-restore.sh
fi

kube::util::ensure-temp-dir
TMP_GOPATH="${KUBE_TEMP}/go"

function updateGodepManifest() {
  pushd "${TMP_GOPATH}/src/k8s.io/${repo}" >/dev/null
    kube::log::status "Updating godeps for k8s.io/${repo}"
    rm -rf Godeps # remove the current Godeps.json so we always rebuild it
    GOPATH="${TMP_GOPATH}:${GOPATH}:${GOPATH}/src/k8s.io/kubernetes/staging" godep save ${GODEP_OPTS} ./... 2>&1 | sed 's/^/  /'

    # Rewriting Godeps.json to remove commits that don't really exist because we haven't pushed the prereqs yet
    go run "${KUBE_ROOT}/staging/godeps-json-updater.go" --godeps-file="${TMP_GOPATH}/src/k8s.io/${repo}/Godeps/Godeps.json" --override-import-path="k8s.io/${repo}"

    # commit so that following repos do not see this repo as dirty
    git add vendor >/dev/null
    git commit -a -m "Updated Godeps.json" >/dev/null
  popd >/dev/null
}

function diffGodepManifest() {
  local ret=0
  diff --ignore-matching-lines='^\s*\"GoVersion\":' --ignore-matching-line='^\s*\"GodepVersion\":' --ignore-matching-lines='^\s*\"Comment\"' -u "${KUBE_ROOT}/staging/src/k8s.io/${repo}/Godeps/Godeps.json" "${TMP_GOPATH}/src/k8s.io/${repo}/Godeps/Godeps.json" || ret=$?
  if [[ "${ret}" != "0" && "${FAIL_ON_DIFF}" == true ]]; then
    exit ${ret}
  fi
}

# Create a fake git repo for staging to prevent godeps from complaining
pushd "${KUBE_ROOT}" >/dev/null
  git init >/dev/null
  git config --local user.email "nobody@k8s.io"
  git config --local user.name "$0"
  git add . >/dev/null
  git commit -q -m "Snapshot" >/dev/null
popd >/dev/null

# move into staging and save the dependencies for everything in order
mkdir -p "${TMP_GOPATH}/src/k8s.io"
for repo in $(ls ${KUBE_ROOT}/staging/src/k8s.io); do
  cp -a "${KUBE_ROOT}/staging/src/k8s.io/${repo}" "${TMP_GOPATH}/src/k8s.io/"

  pushd "${TMP_GOPATH}/src/k8s.io/${repo}" >/dev/null
    git init >/dev/null
    git config --local user.email "nobody@k8s.io"
    git config --local user.name "$0"
    git add . >/dev/null
    git commit -q -m "Snapshot" >/dev/null
  popd >/dev/null

  updateGodepManifest
  diffGodepManifest

  if [ "${DRY_RUN}" != true ]; then
    cp "${TMP_GOPATH}/src/k8s.io/${repo}/Godeps/Godeps.json" "${KUBE_ROOT}/staging/src/k8s.io/${repo}/Godeps/Godeps.json"
    # Assume Godeps.json is not updated, as the working tree needs to be clean
    # Without this, the script would pause after each staging repo to prompt the
    # user to commit all changes (difficult inside a container). It's safe to
    # ignore this file, as we know it's going to be changing, and we don't copy
    # the git tree back out from the container.
    git update-index --assume-unchanged "${KUBE_ROOT}/staging/src/k8s.io/${repo}/Godeps/Godeps.json"
  fi
done
