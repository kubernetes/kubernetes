#!/bin/bash

# Copyright 2016 The Kubernetes Authors.
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

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"
source "${KUBE_ROOT}/hack/lib/util.sh"

kube::log::status "Ensuring prereqs"
kube::util::ensure_single_dir_gopath
kube::util::ensure_no_staging_repos_in_gopath

kube::util::ensure_godep_version

BACKUP=_tmp/godep-save.$RANDOM
mkdir -p "${BACKUP}"

function kube::godep_save::cleanup() {
    if [[ -d "${BACKUP}/vendor" ]]; then
        kube::log::error "${BACKUP}/vendor exists, restoring it"
        rm -rf vendor
        mv "${BACKUP}/vendor" vendor
    fi
    if [[ -d "${BACKUP}/Godeps" ]]; then
        kube::log::error "${BACKUP}/Godeps exists, restoring it"
        rm -rf Godeps
        mv "${BACKUP}/Godeps" Godeps
    fi
}
kube::util::trap_add kube::godep_save::cleanup EXIT

# Clear old state, but save it in case of error
if [[ -d vendor ]]; then
    mv vendor "${BACKUP}/vendor"
fi
if [[ -d Godeps ]]; then
    mv Godeps "${BACKUP}/Godeps"
fi

# Some things we want in godeps aren't code dependencies, so ./...
# won't pick them up.
REQUIRED_BINS=(
  "github.com/onsi/ginkgo/ginkgo"
  "github.com/jteeuwen/go-bindata/go-bindata"
  "github.com/tools/godep"
  "./..."
)

kube::log::status "Running godep save - this might take a while"
# This uses $(pwd) rather than ${KUBE_ROOT} because KUBE_ROOT will be
# realpath'ed, and godep barfs ("... is not using a known version control
# system") on our staging dirs.
GOPATH="${GOPATH}:$(pwd)/staging" godep save "${REQUIRED_BINS[@]}"

# create a symlink in vendor directory pointing to the staging client. This
# let other packages use the staging client as if it were vendored.
for repo in $(ls staging/src/k8s.io); do
  if [ ! -e "vendor/k8s.io/${repo}" ]; then
    ln -s "../../staging/src/k8s.io/${repo}" "vendor/k8s.io/${repo}"
  fi
done

# Workaround broken symlink in docker repo because godep copies the link, but
# not the target
rm -rf vendor/github.com/docker/docker/project/

kube::log::status "Updating BUILD files"
hack/update-bazel.sh >/dev/null

kube::log::status "Updating LICENSES file"
hack/update-godep-licenses.sh >/dev/null

# Clean up
rm -rf "${BACKUP}"
