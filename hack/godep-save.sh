#!/usr/bin/env bash

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
  "github.com/client9/misspell/cmd/misspell"
  "github.com/cloudflare/cfssl/cmd/cfssl"
  "github.com/cloudflare/cfssl/cmd/cfssljson"
  "github.com/bazelbuild/bazel-gazelle/cmd/gazelle"
  "github.com/kubernetes/repo-infra/kazel"
  "k8s.io/kube-openapi/cmd/openapi-gen"
  "golang.org/x/lint/golint"
  "./..."
)

kube::log::status "Running godep save - this might take a while"
# This uses $(pwd) rather than ${KUBE_ROOT} because KUBE_ROOT will be
# realpath'ed, and godep barfs ("... is not using a known version control
# system") on our staging dirs.
GOPATH="${GOPATH}:$(pwd)/staging" ${KUBE_GODEP:?} save "${REQUIRED_BINS[@]}"

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
# Assume that anything imported through godep doesn't need Bazel to build.
# Prune out any Bazel build files, since these can break the build due to
# missing dependencies that aren't included by godep.
find vendor/ -type f \( -name BUILD -o -name BUILD.bazel -o -name WORKSPACE \) \
  -exec rm -f {} \;
hack/update-bazel.sh >/dev/null

kube::log::status "Updating LICENSES file"
hack/update-godep-licenses.sh >/dev/null

kube::log::status "Creating OWNERS file"
rm -f "Godeps/OWNERS" "vendor/OWNERS"
cat <<__EOF__ > "Godeps/OWNERS"
approvers:
- dep-approvers
__EOF__
cp "Godeps/OWNERS" "vendor/OWNERS"

# Clean up
rm -rf "${BACKUP}"
