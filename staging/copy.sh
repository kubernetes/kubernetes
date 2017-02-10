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

FAIL_ON_CHANGES=false
DRY_RUN=false
while getopts ":fd" opt; do
  case $opt in
    f)
      FAIL_ON_CHANGES=true
      ;;
    d)
      DRY_RUN=true
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done
readonly FAIL_ON_CHANGES DRY_RUN

echo "**PLEASE** run \"godep restore\" before running this script"
# PREREQUISITES: run `godep restore` in the main repo before calling this script.
CLIENTSET="clientset"
MAIN_REPO_FROM_SRC="k8s.io/kubernetes"
MAIN_REPO="${GOPATH%:*}/src/${MAIN_REPO_FROM_SRC}"
CLIENT_REPO_FROM_SRC="k8s.io/client-go"
CLIENT_REPO_TEMP_FROM_SRC="k8s.io/_tmp"
CLIENT_REPO="${MAIN_REPO}/staging/src/${CLIENT_REPO_FROM_SRC}"
CLIENT_REPO_TEMP="${MAIN_REPO}/staging/src/${CLIENT_REPO_TEMP_FROM_SRC}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if LANG=C sed --help 2>&1 | grep -q GNU; then
  SED="sed"
elif which gsed &>/dev/null; then
  SED="gsed"
else
  echo "Failed to find GNU sed as sed or gsed. If you are on Mac: brew install gnu-sed." >&2
  exit 1
fi

cleanup() {
    rm -rf "${CLIENT_REPO_TEMP}"
}

trap cleanup EXIT SIGINT

# working in the ${CLIENT_REPO_TEMP} so 'godep save' won't complain about dirty working tree.
echo "creating the tmp directory"
mkdir -p "${CLIENT_REPO_TEMP}"
cd "${CLIENT_REPO}"

# there are two classes of package in staging/client-go, those which are authoritative (client-go has the only copy)
# and those which are copied and rewritten (client-go is not authoritative).
# we first copy out the authoritative packages to the temp location, then copy non-authoritative packages
# then save over the original


# save copies code from client-go into the temp folder to make sure we don't lose it by accident
# TODO this is temporary until everything in certain directories is authoritative
function save() {
    mkdir -p "${CLIENT_REPO_TEMP}/$1"
    cp -r "${CLIENT_REPO}/$1/"* "${CLIENT_REPO_TEMP}/$1"
}

# save everything for which the staging directory is the source of truth
save "discovery"
save "dynamic"
save "rest"
save "testing"
save "tools"
save "transport"
save "third_party"
save "plugin"
save "util"



# mkcp copies file from the main repo to the client repo, it creates the directory if it doesn't exist in the client repo.
function mkcp() {
    mkdir -p "${CLIENT_REPO_TEMP}/$2" && cp -r "${MAIN_REPO}/$1" "${CLIENT_REPO_TEMP}/$2"
}

# assemble all the other parts of the staging directory
echo "copying client packages"
# need to copy version.  We aren't authoritative here
# version has subdirs which we don't need.  Only copy the files we want
mkdir -p "${CLIENT_REPO_TEMP}/pkg/version"
find "${MAIN_REPO}/pkg/version" -maxdepth 1 -type f | xargs -I{} cp {} "${CLIENT_REPO_TEMP}/pkg/version"
# need to copy clientsets, though later we should copy APIs and later generate clientsets
mkcp "pkg/client/clientset_generated/${CLIENTSET}" "pkg/client/clientset_generated"

pushd "${CLIENT_REPO_TEMP}" > /dev/null
echo "generating vendor/"
# client-go depends on some apimachinery packages. Adding staging/ to the GOPATH
# so that if client-go has new dependencies on apimachinery, `godep save` can
# find the dependent packages in staging/, instead of failing. Note that all
# k8s.io/apimachinery dependencies will be updated later by the robot to point
# to the real k8s.io/apimachinery commit and vendor the real code.
GOPATH="${GOPATH}:${MAIN_REPO}/staging"
GO15VENDOREXPERIMENT=1 godep save ./...
popd > /dev/null

echo "moving vendor/k8s.io/kubernetes"
cp -r "${CLIENT_REPO_TEMP}"/vendor/k8s.io/kubernetes/* "${CLIENT_REPO_TEMP}"/
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
    xargs ${SED} -i "s|\"${MAIN_REPO_FROM_SRC}|\"${CLIENT_REPO_FROM_SRC}|g"

echo "rewrite proto names in proto.RegisterType"
find "${CLIENT_REPO_TEMP}" -type f -name "generated.pb.go" -print0 | xargs -0 ${SED} -i "s/k8s\.io\.kubernetes/k8s.io.client-go/g"

# strip all generator tags from client-go
find "${CLIENT_REPO_TEMP}" -type f -name "*.go" -print0 | xargs -0 ${SED} -i '/^\/\/ +k8s:openapi-gen=true/d'
find "${CLIENT_REPO_TEMP}" -type f -name "*.go" -print0 | xargs -0 ${SED} -i '/^\/\/ +k8s:defaulter-gen=/d'
find "${CLIENT_REPO_TEMP}" -type f -name "*.go" -print0 | xargs -0 ${SED} -i '/^\/\/ +k8s:deepcopy-gen=/d'
find "${CLIENT_REPO_TEMP}" -type f -name "*.go" -print0 | xargs -0 ${SED} -i '/^\/\/ +k8s:conversion-gen=/d'


echo "rearranging directory layout"
# $1 and $2 are relative to ${CLIENT_REPO_TEMP}
function mvfolder {
    local src=${1%/#/}
    local dst=${2%/#/}
    mkdir -p "${CLIENT_REPO_TEMP}/${dst}"
    # move
    mv "${CLIENT_REPO_TEMP}/${src}"/* "${CLIENT_REPO_TEMP}/${dst}"
    # rewrite package
    local src_package="${src##*/}"
    local dst_package="${dst##*/}"
    find "${CLIENT_REPO_TEMP}/${dst}" -type f -name "*.go" -print0 | xargs -0 ${SED} -i "s,package ${src_package},package ${dst_package},g"

    { grep -Rl "\"${CLIENT_REPO_FROM_SRC}/${src}" "${CLIENT_REPO_TEMP}" || true ; } | while read -r target ; do
        # rewrite imports
        # the first rule is to convert import lines like `restclient "k8s.io/client-go/pkg/client/restclient"`,
        # where a package alias is the same the package name.
        ${SED} -i "s,\<${src_package} \"${CLIENT_REPO_FROM_SRC}/${src},${dst_package} \"${CLIENT_REPO_FROM_SRC}/${dst},g" "${target}"
        ${SED} -i "s,\"${CLIENT_REPO_FROM_SRC}/${src},\"${CLIENT_REPO_FROM_SRC}/${dst},g" "${target}"
        # rewrite import invocation
        if [ "${src_package}" != "${dst_package}" ]; then
            ${SED} -i "s,\<${src_package}\.\([a-zA-Z]\),${dst_package}\.\1,g" "${target}"
        fi
    done
}

mvfolder "pkg/client/clientset_generated/${CLIENTSET}" kubernetes
if [ "$(find "${CLIENT_REPO_TEMP}"/pkg/client -type f -name "*.go")" ]; then
    echo "${CLIENT_REPO_TEMP}/pkg/client is expected to be empty"
    exit 1
else
    rm -r "${CLIENT_REPO_TEMP}"/pkg/client
fi

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

echo "remove cyclical godep"
rm -rf "${CLIENT_REPO_TEMP}/_vendor/k8s.io/client-go"
# If godep cannot find dependent packages in the primary GOPATH, it will search
# in the secondary GOPATH staging/. If successful, godep will wrongly copy the
# dependents relative to the primary GOPATH (looks like a godep bug), creating
# the ${CLIENT_REPO_TEMP}/staging dir. These copies will not be recognized by
# the Go compiler, so we just remove them. Note that the publishing robot will
# correctly resolve these dependencies later.
rm -rf "${CLIENT_REPO_TEMP}/staging"

if [ "${FAIL_ON_CHANGES}" = true ]; then
    echo "running FAIL_ON_CHANGES"
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

# clean the ${CLIENT_REPO}
echo "move to the client repo"
if [ "${DRY_RUN}" = false ]; then
    ls "${CLIENT_REPO}" | { grep -v '_tmp' || true; } | xargs rm -rf
    mv "${CLIENT_REPO_TEMP}"/* "${CLIENT_REPO}"
fi

cleanup
