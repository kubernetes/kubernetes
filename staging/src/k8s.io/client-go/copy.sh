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

# PREREQUISITES: run `godep restore` in the main repo before calling this script.
RELEASE="1.4"
MAIN_REPO_FROM_SRC="${1:-"k8s.io/kubernetes"}"
MAIN_REPO="${GOPATH%:*}/src/${MAIN_REPO_FROM_SRC}"
CLIENT_REPO_FROM_SRC="${2:-"k8s.io/client-go/${RELEASE}"}"
CLIENT_REPO="${MAIN_REPO}/staging/src/${CLIENT_REPO_FROM_SRC}"
CLIENT_REPO_TEMP="${CLIENT_REPO}"/_tmp

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# working in the ${CLIENT_REPO_TEMP} so 'godep save' won't complain about dirty working tree.
echo "creating the _tmp directory"
mkdir -p "${CLIENT_REPO_TEMP}"
cd "${CLIENT_REPO}"

# mkcp copies file from the main repo to the client repo, it creates the directory if it doesn't exist in the client repo.
function mkcp() {
    mkdir -p "${CLIENT_REPO_TEMP}/$2" && cp -r "${MAIN_REPO}/$1" "${CLIENT_REPO_TEMP}/$2"
}

echo "copying client packages"
mkcp "pkg/client/clientset_generated/release_1_4" "pkg/client/clientset_generated"
mkcp "/pkg/client/record/" "/pkg/client"
mkcp "/pkg/client/cache/" "/pkg/client"
# TODO: make this test file not depending on pkg/client/unversioned
rm "${CLIENT_REPO_TEMP}"/pkg/client/cache/listwatch_test.go
mkcp "/pkg/client/restclient" "/pkg/client"
mkcp "/pkg/client/testing" "/pkg/client"
# remove this test because it imports the internal clientset
rm "${CLIENT_REPO_TEMP}"/pkg/client/testing/core/fake_test.go
mkcp "/pkg/client/transport" "/pkg/client"
mkcp "/pkg/client/typed" "/pkg/client"

mkcp "/pkg/client/unversioned/auth" "/pkg/client/unversioned"
mkcp "/pkg/client/unversioned/clientcmd" "/pkg/client/unversioned"
mkcp "/pkg/client/unversioned/portforward" "/pkg/client/unversioned"
# remove this test because it imports the internal clientset
rm "${CLIENT_REPO_TEMP}"/pkg/client/unversioned/portforward/portforward_test.go

pushd "${CLIENT_REPO_TEMP}" > /dev/null
echo "generating vendor/"
GO15VENDOREXPERIMENT=1 godep save ./...
popd > /dev/null

echo "move to the client repo"
# clean the ${CLIENT_REPO}
ls "${CLIENT_REPO}" | grep -v '_tmp' | xargs rm -r
mv "${CLIENT_REPO_TEMP}"/* "${CLIENT_REPO}"
rm -r "${CLIENT_REPO_TEMP}"

echo "moving vendor/k8s.io/kuberentes"
cp -rn "${CLIENT_REPO}"/vendor/k8s.io/kubernetes/. "${CLIENT_REPO}"/
rm -rf "${CLIENT_REPO}"/vendor/k8s.io/kubernetes
# client-go will share the vendor of the main repo for now. When client-go
# becomes a standalone repo, it will have its own vendor
mv "${CLIENT_REPO}"/vendor "${CLIENT_REPO}"/_vendor
# remove the pkg/util/net/sets/README.md to silent hack/verify-munge-docs.sh
# TODO: probably we should convert the README.md a doc.go
find ./ -name "README.md" -delete

echo "rewriting Godeps.json"
go run "${DIR}/godeps-json-updater.go" --godeps-file="${CLIENT_REPO}/Godeps/Godeps.json" --client-go-import-path="${CLIENT_REPO_FROM_SRC}"

echo "rewriting imports"
grep -Rl "\"${MAIN_REPO_FROM_SRC}" ./ | grep ".go" | grep -v "vendor/" | xargs sed -i "s|\"${MAIN_REPO_FROM_SRC}|\"${CLIENT_REPO_FROM_SRC}|g"

echo "converting pkg/client/record to v1"
# need a v1 version of ref.go
cp "${CLIENT_REPO}"/pkg/api/ref.go "${CLIENT_REPO}"/pkg/api/v1/ref.go
gofmt -w -r 'api.a -> v1.a' "${CLIENT_REPO}"/pkg/api/v1/ref.go
gofmt -w -r 'Scheme -> api.Scheme' "${CLIENT_REPO}"/pkg/api/v1/ref.go
# rewriting package name to v1
sed -i 's/package api/package v1/g' "${CLIENT_REPO}"/pkg/api/v1/ref.go
# ref.go refers api.Scheme, so manually import /pkg/api
sed -i "s,import (,import (\n\"${CLIENT_REPO_FROM_SRC}/pkg/api\",g" "${CLIENT_REPO}"/pkg/api/v1/ref.go
gofmt -w "${CLIENT_REPO}"/pkg/api/v1/ref.go 
# rewrite pkg/client/record to v1
gofmt -w -r 'api.a -> v1.a' "${CLIENT_REPO}"/pkg/client/record
# need to call sed to rewrite the strings in test cases...
find "${CLIENT_REPO}"/pkg/client/record -type f -name "*.go" -print0 | xargs -0 sed -i "s/api.ObjectReference/v1.ObjectReference/g"
# rewrite the imports
find "${CLIENT_REPO}"/pkg/client/record -type f -name "*.go" -print0 | xargs -0 sed -i 's,pkg/api",pkg/api/v1",g'
# gofmt the changed files

echo "rewrite conflicting Prometheus registration"
sed -i "s/request_latency_microseconds/request_latency_microseconds_copy/g" "${CLIENT_REPO}"/pkg/client/metrics/metrics.go
sed -i "s/request_status_codes/request_status_codes_copy/g" "${CLIENT_REPO}"/pkg/client/metrics/metrics.go
sed -i "s/kubernetes_build_info/kubernetes_build_info_copy/g" "${CLIENT_REPO}"/pkg/version/version.go

echo "rewrite proto names in proto.RegisterType"
find "${CLIENT_REPO}" -type f -name "generated.pb.go" -print0 | xargs -0 sed -i "s/k8s\.io\.kubernetes/k8s.io.client-go.1.4/g"

echo "rearranging directory layout"
# $1 and $2 are relative to ${CLIENT_REPO}
function mvfolder {
    local src=${1%/#/}
    local dst=${2%/#/}
    # create the parent directory of dst
    if [ "${dst%/*}" != "${dst}" ]; then
        mkdir -p "${CLIENT_REPO}/${dst%/*}"
    fi
    # move
    mv "${CLIENT_REPO}/${src}" "${CLIENT_REPO}/${dst}"
    # rewrite package
    local src_package="${src##*/}"
    local dst_package="${dst##*/}"
    find "${CLIENT_REPO}" -type f -name "*.go" -print0 | xargs -0 sed -i "s,package ${src_package},package ${dst_package},g"
    # rewrite imports
    # the first rule is to convert import lines like `restclient "k8s.io/client-go/pkg/client/restclient"`,
    # where a package alias is the same the package name.
    find "${CLIENT_REPO}" -type f -name "*.go" -print0 | \
        xargs -0 sed -i "s,${src_package} \"${CLIENT_REPO_FROM_SRC}/${src},${dst_package} \"${CLIENT_REPO_FROM_SRC}/${dst},g"
    find "${CLIENT_REPO}" -type f -name "*.go" -print0 | \
        xargs -0 sed -i "s,\"${CLIENT_REPO_FROM_SRC}/${src},\"${CLIENT_REPO_FROM_SRC}/${dst},g"
    # rewrite import invocation
    if [ "${src_package}" != "${dst_package}" ]; then
        find "${CLIENT_REPO}" -type f -name "*.go" -print0 | xargs -0 sed -i "s,\<${src_package}\.\([a-zA-Z]\),${dst_package}\.\1,g"
    fi
}

mvfolder pkg/client/clientset_generated/release_1_4 kubernetes
mvfolder pkg/client/typed/discovery discovery
mvfolder pkg/client/typed/dynamic dynamic
mvfolder pkg/client/transport transport
mvfolder pkg/client/record tools/record
mvfolder pkg/client/restclient rest
mvfolder pkg/client/cache tools/cache
mvfolder pkg/client/unversioned/auth tools/auth
mvfolder pkg/client/unversioned/clientcmd tools/clientcmd
mvfolder pkg/client/unversioned/portforward tools/portforward
mvfolder pkg/client/metrics tools/metrics
mvfolder pkg/client/testing/core testing
if [ "$(find "${CLIENT_REPO}"/pkg/client -type f -name "*.go")" ]; then
    echo "${CLIENT_REPO}/pkg/client is expected to be empty"
    exit 1
else 
    rm -r "${CLIENT_REPO}"/pkg/client
fi
mvfolder third_party pkg/third_party
mvfolder federation pkg/federation

echo "running gofmt"
find "${CLIENT_REPO}" -type f -name "*.go" -print0 | xargs -0 gofmt -w
