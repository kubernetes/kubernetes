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

export KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::util::ensure-gnu-sed

# Remove generated files prior to running kazel.
# TODO(spxtr): Remove this line once Bazel is the only way to build.
rm -f "${KUBE_ROOT}/pkg/generated/openapi/zz_generated.openapi.go"

# Ensure that we find the binaries we build before anything else.
export GOBIN="${KUBE_OUTPUT_BINPATH}"
PATH="${GOBIN}:${PATH}"

# Install tools we need, but only from vendor/...
go install ./vendor/github.com/bazelbuild/bazel-gazelle/cmd/gazelle

go install ./vendor/github.com/kubernetes/repo-infra/kazel

touch "${KUBE_ROOT}/vendor/BUILD"

gazelle-fix() {

gazelle fix \
    -build_file_name=BUILD,BUILD.bazel \
    -external=vendored \
    -proto=legacy \
    -mode=fix
# gazelle gets confused by our staging/ directory, prepending an extra
# "k8s.io/kubernetes/staging/src" to the import path.
# gazelle won't follow the symlinks in vendor/, so we can't just exclude
# staging/. Instead we just fix the bad paths with sed.
find staging -name BUILD -o -name BUILD.bazel | \
  xargs ${SED} -i 's|\(importpath = "\)k8s.io/kubernetes/staging/src/\(.*\)|\1\2|'
}

gazelle-fix

kazel

cd "${KUBE_ROOT}"
go install github.com/bazelbuild/buildtools/buildozer
echo looking for needs $KUBE_ROOT...
want=$(find . -iname *.go | (xargs grep -l '+k8s:deepcopy-gen=' || true) | xargs -n 1 dirname | sort -u | sed -e 's|^./||')
echo silly files $want
deepcopies="$(find . -iname zz_generated.deepcopy.go | xargs -n 1 dirname | sort -u | sed -e 's|^./||')"
have="$(find . -name BUILD -or -name BUILD.bazel | xargs grep -l k8s_deepcopy)"
echo Deleting existing k8s_deepcopy commands... $have
echo $have | xargs sed -e '/^k8s_deepcopy/d' -i ''
echo "Adding k8s_deepcopy() rule"
for w in $want; do
  if [[ $w == "vendor/k8s.io/code-generator/cmd/deepcopy-gen" || \
        $w == "staging/src/k8s.io/code-generator/cmd/deepcopy-gen" ]]; then
	  echo ignoring deepcopy-gen binary
	  continue
  fi
  if [[ -f $w/BUILD.bazel ]]; then
	  echo 'k8s_deepcopy(outs=["zz_generated.deepcopy.go"])' >> $w/BUILD
	  buildozer 'new_load //build:deepcopy.bzl k8s_deepcopy' //$w:__pkg__
  elif  [[ -f $w/BUILD ]]; then
	  echo 'k8s_deepcopy(outs=["zz_generated.deepcopy.go"])' >> $w/BUILD
  else
	  echo cannot find build file for $w
	  continue
  fi
  buildozer 'new_load //build:deepcopy.bzl k8s_deepcopy' //$w:__pkg__
done
echo Deleting zz_generated.deepcopy.go files
echo $deepcopies | xargs -n 1 -I '{}' rm {}/zz_generated.deepcopy.go
gazelle-fix
