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

VERIFYONLY=false
while getopts ":v" opt; do
  case $opt in
    v)
      VERIFYONLY=true
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done
readonly VERIFYONLY

echo "**PLEASE** run \"godep restore\" before running this script"
# PREREQUISITES: run `godep restore` in the main repo before calling this script.
CLIENTSET="release_1_5"
MAIN_REPO_FROM_SRC="k8s.io/kubernetes"
MAIN_REPO="${GOPATH%:*}/src/${MAIN_REPO_FROM_SRC}"
CLIENT_REPO_FROM_SRC="k8s.io/client-go"
CLIENT_REPO_TEMP_FROM_SRC="k8s.io/_tmp"
CLIENT_REPO="${MAIN_REPO}/staging/src/${CLIENT_REPO_FROM_SRC}"
CLIENT_REPO_TEMP="${MAIN_REPO}/staging/src/${CLIENT_REPO_TEMP_FROM_SRC}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cleanup() {
    rm -rf "${CLIENT_REPO_TEMP}"
}

trap cleanup EXIT SIGINT

# working in the ${CLIENT_REPO_TEMP} so 'godep save' won't complain about dirty working tree.
echo "creating the tmp directory"
mkdir -p "${CLIENT_REPO_TEMP}"
cd "${CLIENT_REPO}"

# mkcp copies file from the main repo to the client repo, it creates the directory if it doesn't exist in the client repo.
function mkcp() {
    mkdir -p "${CLIENT_REPO_TEMP}/$2" && cp -r "${MAIN_REPO}/$1" "${CLIENT_REPO_TEMP}/$2"
}

echo "copying client packages"
mkcp "pkg/client/clientset_generated/${CLIENTSET}" "pkg/client/clientset_generated"
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

mkcp "/plugin/pkg/client/auth" "/plugin/pkg/client"
# remove this test because it imports the internal clientset
rm "${CLIENT_REPO_TEMP}"/pkg/client/unversioned/portforward/portforward_test.go

pushd "${CLIENT_REPO_TEMP}" > /dev/null
echo "generating vendor/"
GO15VENDOREXPERIMENT=1 godep save ./...
popd > /dev/null

echo "moving vendor/k8s.io/kuberentes"
cp -rn "${CLIENT_REPO_TEMP}"/vendor/k8s.io/kubernetes/. "${CLIENT_REPO_TEMP}"/
rm -rf "${CLIENT_REPO_TEMP}"/vendor/k8s.io/kubernetes
# client-go will share the vendor of the main repo for now. When client-go
# becomes a standalone repo, it will have its own vendor
mv "${CLIENT_REPO_TEMP}"/vendor "${CLIENT_REPO_TEMP}"/_vendor

echo "rewriting Godeps.json"
go run "${DIR}/godeps-json-updater.go" --godeps-file="${CLIENT_REPO_TEMP}/Godeps/Godeps.json" --client-go-import-path="${CLIENT_REPO_FROM_SRC}"

echo "rewriting imports"
grep -Rl "\"${MAIN_REPO_FROM_SRC}" "${CLIENT_REPO_TEMP}" | \
    grep "\.go" | \
    grep -v "vendor/" | \
    xargs sed -i "s|\"${MAIN_REPO_FROM_SRC}|\"${CLIENT_REPO_FROM_SRC}|g"

echo "converting pkg/client/record to v1"
# need a v1 version of ref.go
cp "${CLIENT_REPO_TEMP}"/pkg/api/ref.go "${CLIENT_REPO_TEMP}"/pkg/api/v1/ref.go
gofmt -w -r 'api.a -> v1.a' "${CLIENT_REPO_TEMP}"/pkg/api/v1/ref.go
gofmt -w -r 'Scheme -> api.Scheme' "${CLIENT_REPO_TEMP}"/pkg/api/v1/ref.go
# rewriting package name to v1
sed -i 's/package api/package v1/g' "${CLIENT_REPO_TEMP}"/pkg/api/v1/ref.go
# ref.go refers api.Scheme, so manually import /pkg/api
sed -i "s,import (,import (\n\"${CLIENT_REPO_FROM_SRC}/pkg/api\",g" "${CLIENT_REPO_TEMP}"/pkg/api/v1/ref.go
gofmt -w "${CLIENT_REPO_TEMP}"/pkg/api/v1/ref.go 
# rewrite pkg/client/record to v1
gofmt -w -r 'api.a -> v1.a' "${CLIENT_REPO_TEMP}"/pkg/client/record
# need to call sed to rewrite the strings in test cases...
find "${CLIENT_REPO_TEMP}"/pkg/client/record -type f -name "*.go" -print0 | xargs -0 sed -i "s/api.ObjectReference/v1.ObjectReference/g"
# rewrite the imports
find "${CLIENT_REPO_TEMP}"/pkg/client/record -type f -name "*.go" -print0 | xargs -0 sed -i 's,pkg/api",pkg/api/v1",g'
# gofmt the changed files

echo "rewrite conflicting Prometheus registration"
sed -i "s/request_latency_microseconds/request_latency_microseconds_copy/g" "${CLIENT_REPO_TEMP}"/pkg/client/metrics/metrics.go
sed -i "s/request_status_codes/request_status_codes_copy/g" "${CLIENT_REPO_TEMP}"/pkg/client/metrics/metrics.go
sed -i "s/kubernetes_build_info/kubernetes_build_info_copy/g" "${CLIENT_REPO_TEMP}"/pkg/version/version.go

echo "rewrite proto names in proto.RegisterType"
find "${CLIENT_REPO_TEMP}" -type f -name "generated.pb.go" -print0 | xargs -0 sed -i "s/k8s\.io\.kubernetes/k8s.io.client-go/g"

echo "rearranging directory layout"
# $1 and $2 are relative to ${CLIENT_REPO_TEMP}
function mvfolder {
    local src=${1%/#/}
    local dst=${2%/#/}
    # create the parent directory of dst
    if [ "${dst%/*}" != "${dst}" ]; then
        mkdir -p "${CLIENT_REPO_TEMP}/${dst%/*}"
    fi
    # move
    mv "${CLIENT_REPO_TEMP}/${src}" "${CLIENT_REPO_TEMP}/${dst}"
    # rewrite package
    local src_package="${src##*/}"
    local dst_package="${dst##*/}"
    find "${CLIENT_REPO_TEMP}/${dst}" -type f -name "*.go" -print0 | xargs -0 sed -i "s,package ${src_package},package ${dst_package},g"

    { grep -Rl "\"${CLIENT_REPO_FROM_SRC}/${src}" "${CLIENT_REPO_TEMP}" || true ; } | while read -r target ; do
        # rewrite imports
        # the first rule is to convert import lines like `restclient "k8s.io/client-go/pkg/client/restclient"`,
        # where a package alias is the same the package name.
        sed -i "s,\<${src_package} \"${CLIENT_REPO_FROM_SRC}/${src},${dst_package} \"${CLIENT_REPO_FROM_SRC}/${dst},g" "${target}"
        sed -i "s,\"${CLIENT_REPO_FROM_SRC}/${src},\"${CLIENT_REPO_FROM_SRC}/${dst},g" "${target}"
        # rewrite import invocation
        if [ "${src_package}" != "${dst_package}" ]; then
            sed -i "s,\<${src_package}\.\([a-zA-Z]\),${dst_package}\.\1,g" "${target}"
        fi
    done
}

mvfolder "pkg/client/clientset_generated/${CLIENTSET}" kubernetes
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
mvfolder pkg/client/testing/cache tools/cache/testing
mvfolder cmd/kubeadm/app/apis/kubeadm pkg/apis/kubeadm
if [ "$(find "${CLIENT_REPO_TEMP}"/pkg/client -type f -name "*.go")" ]; then
    echo "${CLIENT_REPO_TEMP}/pkg/client is expected to be empty"
    exit 1
else 
    rm -r "${CLIENT_REPO_TEMP}"/pkg/client
fi
mvfolder third_party pkg/third_party
mvfolder federation pkg/federation

echo "running gofmt"
find "${CLIENT_REPO_TEMP}" -type f -name "*.go" -print0 | xargs -0 gofmt -w

echo "remove black listed files"
find "${CLIENT_REPO_TEMP}" -type f \( \
    -name "*BUILD" -o \
    -name "*.json" -not -name "Godeps.json" -o \
    -name "*.yaml" -o \
    -name "*.yml" -o \
    -name "*.sh" \
    \) -delete

if [ "${VERIFYONLY}" = true ]; then
    echo "running verify-only"
    ret=0
    if diff -NauprB -I "GoVersion.*\|GodepVersion.*" "${CLIENT_REPO}" "${CLIENT_REPO_TEMP}"; then
      echo "${CLIENT_REPO} up to date."
      cleanup
      exit 0
    else
      echo "${CLIENT_REPO} is out of date. Please run hack/update-client-go.sh"
      cleanup
      exit 1
    fi
fi

echo "move to the client repo"
# clean the ${CLIENT_REPO}
ls "${CLIENT_REPO}" | { grep -v '_tmp' || true; } | xargs rm -rf
mv "${CLIENT_REPO_TEMP}"/* "${CLIENT_REPO}"
cleanup
