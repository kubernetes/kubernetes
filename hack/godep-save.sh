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

## Workaround "httplex" was dropped from golang.org/x/net repo and the code
## was moved to the "golang.org/x/net/http/httpguts" directory, we do not use
## this directly, however many packages we vendor are still using the older
## golang.org/x/net and we need to keep this until all those dependencies
## are switched to newer golang.org/x/net.
IGNORED_PACKAGES=(
  "golang.org/x/net/lex/httplex"
)
REQUIRED_BINS=(
  "golang.org/x/net/internal/nettest"
  "golang.org/x/net/internal/socks"
  "golang.org/x/net/internal/sockstest"
)

# Some things we want in godeps aren't code dependencies, so ./...
# won't pick them up.
REQUIRED_BINS+=(
  "github.com/bazelbuild/bazel-gazelle/cmd/gazelle"
  "github.com/bazelbuild/buildtools/buildozer"
  "github.com/cespare/prettybench"
  "github.com/client9/misspell/cmd/misspell"
  "github.com/cloudflare/cfssl/cmd/cfssl"
  "github.com/cloudflare/cfssl/cmd/cfssljson"
  "github.com/jstemmer/go-junit-report"
  "github.com/jteeuwen/go-bindata/go-bindata"
  "github.com/onsi/ginkgo/ginkgo"
  "golang.org/x/lint/golint"
  "k8s.io/kube-openapi/cmd/openapi-gen"
  "k8s.io/repo-infra/kazel"
  "./..."
)

kube::log::status "Running godep save - this might take a while"
# This uses $(pwd) rather than ${KUBE_ROOT} because KUBE_ROOT will be
# realpath'ed, and godep barfs ("... is not using a known version control
# system") on our staging dirs.
GOPATH="${GOPATH}:$(pwd)/staging" ${KUBE_GODEP:?} save -i $(IFS=,; echo "${IGNORED_PACKAGES[*]}") "${REQUIRED_BINS[@]}"

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
# See the OWNERS docs at https://go.k8s.io/owners

approvers:
- dep-approvers
__EOF__
cp "Godeps/OWNERS" "vendor/OWNERS"

# Clean up
rm -rf "${BACKUP}"
