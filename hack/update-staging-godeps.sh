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
# TODO this does not address client-go, since it takes a different approach to vendoring
# TODO client-go should probably be made consistent

set -o errexit
set -o nounset
set -o pipefail

V=""
DRY_RUN=false
FAIL_ON_DIFF=false
while getopts ":vdf" opt; do
  case $opt in
    v) # increase verbosity
      V="-v"
      ;;
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
readonly V IN_PLACE

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"
source "${KUBE_ROOT}/hack/lib/util.sh"

echo "Checking whether godeps are restored"
if ! kube::util::godep_restored 2>&1 | sed 's/^/  /'; then
  echo -e '\nExecute script 'hack/godep-restore.sh' to download dependencies.' 1>&2
  exit 1
fi

kube::util::ensure-temp-dir
kube::util::ensure_godep_version v79

TMP_GOPATH="${KUBE_TEMP}/go"

function updateGodepManifest() {
  local repo="${1}"
  pushd "${TMP_GOPATH}/src/k8s.io/${repo}" >/dev/null
    echo "Updating godeps for k8s.io/${repo}"
    rm -rf Godeps # remove the current Godeps.json so we always rebuild it
    GOPATH="${TMP_GOPATH}:${GOPATH}:${KUBE_ROOT}/staging" godep save ${V} ./... 2>&1 | sed 's/^/  /'

    echo "Rewriting Godeps.json to remove commits that don't really exist because we haven't pushed the prereqs yet"
    go run "${KUBE_ROOT}/staging/godeps-json-updater.go" --godeps-file="${TMP_GOPATH}/src/k8s.io/${repo}/Godeps/Godeps.json" --override-import-path="k8s.io/${repo}"

    # commit so that following repos do not see this repo as dirty
    git add vendor >/dev/null
    git commit -a -m "Updated Godeps.json" >/dev/null
  popd >/dev/null
}

# move into staging and save the dependencies for everything in order
mkdir -p "${TMP_GOPATH}/src/k8s.io"
for repo in $(ls ${KUBE_ROOT}/staging/src/k8s.io); do
  # we have to skip client-go because it does unusual manipulation of its godeps
  if [ "${repo}" == "client-go" ]; then
    continue
  fi
  # we skip metrics because it's synced to the real repo manually
  if [ "${repo}" == "metrics" ]; then
    continue
  fi

  kube::util::ensure_clean_working_dir

  cp -a "${KUBE_ROOT}/staging/src/k8s.io/${repo}" "${TMP_GOPATH}/src/k8s.io/"

  pushd "${TMP_GOPATH}/src/k8s.io/${repo}" >/dev/null
    git init >/dev/null
    git config --local user.email "nobody@k8s.io"
    git config --local user.name "$0"
    git add . >/dev/null
    git commit -q -m "Snapshot" >/dev/null
  popd >/dev/null

  updateGodepManifest "${repo}"

  if [ "${FAIL_ON_DIFF}" == true ]; then
    diff --ignore-matching-lines='^\s*\"GoVersion\":' --ignore-matching-line='^\s*\"GodepVersion\":' --ignore-matching-lines='^\s*\"Comment\"' -u "${KUBE_ROOT}/staging/src/k8s.io/${repo}/Godeps" "${TMP_GOPATH}/src/k8s.io/${repo}/Godeps/Godeps.json"
  fi
  if [ "${DRY_RUN}" != true ]; then
    cp "${TMP_GOPATH}/src/k8s.io/${repo}/Godeps/Godeps.json" "${KUBE_ROOT}/staging/src/k8s.io/${repo}/Godeps"
  fi
done
