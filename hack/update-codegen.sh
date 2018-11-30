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

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env

BUILD_TARGETS=(
  vendor/k8s.io/code-generator/cmd/client-gen
  vendor/k8s.io/code-generator/cmd/lister-gen
  vendor/k8s.io/code-generator/cmd/informer-gen
)
make -C "${KUBE_ROOT}" WHAT="${BUILD_TARGETS[*]}"

clientgen=$(kube::util::find-binary "client-gen")
listergen=$(kube::util::find-binary "lister-gen")
informergen=$(kube::util::find-binary "informer-gen")

# Please do not add any logic to this shell script. Add logic to the go code
# that generates the set-gen program.
#

GROUP_VERSIONS=(${KUBE_AVAILABLE_GROUP_VERSIONS})
GV_DIRS=()
INTERNAL_DIRS=()
for gv in "${GROUP_VERSIONS[@]}"; do
  # add items, but strip off any leading apis/ you find to match command expectations
  api_dir=$(kube::util::group-version-to-pkg-path "${gv}")
  nopkg_dir=${api_dir#pkg/}
  nopkg_dir=${nopkg_dir#vendor/k8s.io/api/}
  pkg_dir=${nopkg_dir#apis/}


  # skip groups that aren't being served, clients for these don't matter
    if [[ " ${KUBE_NONSERVER_GROUP_VERSIONS} " == *" ${gv} "* ]]; then
      continue
    fi

  GV_DIRS+=("${pkg_dir}")

  # collect internal groups
  int_group="${pkg_dir%/*}/"
  if [[ "${pkg_dir}" = core/* ]]; then
    int_group="api/"
  fi
    if ! [[ " ${INTERNAL_DIRS[@]:-} " =~ " ${int_group} " ]]; then
      INTERNAL_DIRS+=("${int_group}")
    fi
done
# delimit by commas for the command
GV_DIRS_CSV=$(IFS=',';echo "${GV_DIRS[*]// /,}";IFS=$)
INTERNAL_DIRS_CSV=$(IFS=',';echo "${INTERNAL_DIRS[*]// /,}";IFS=$)

# This can be called with one flag, --verify-only, so it works for both the
# update- and verify- scripts.
${clientgen} --input-base="k8s.io/kubernetes/pkg/apis" --input="${INTERNAL_DIRS_CSV}" "$@"
${clientgen} --output-base "${KUBE_ROOT}/vendor" --output-package="k8s.io/client-go" --clientset-name="kubernetes" --input-base="k8s.io/kubernetes/vendor/k8s.io/api" --input="${GV_DIRS_CSV}" --go-header-file ${KUBE_ROOT}/hack/boilerplate/boilerplate.generatego.txt "$@"

listergen_external_apis=(
$(
  cd ${KUBE_ROOT}/staging/src
  find k8s.io/api -name types.go | xargs -n1 dirname | sort
)
)
listergen_external_apis_csv=$(IFS=,; echo "${listergen_external_apis[*]}")
${listergen} --output-base "${KUBE_ROOT}/vendor" --output-package "k8s.io/client-go/listers" --input-dirs "${listergen_external_apis_csv}" --go-header-file ${KUBE_ROOT}/hack/boilerplate/boilerplate.generatego.txt "$@"

informergen_external_apis=(
$(
  cd ${KUBE_ROOT}/staging/src
  # because client-gen doesn't do policy/v1alpha1, we have to skip it too
  find k8s.io/api -name types.go | xargs -n1 dirname | sort | grep -v pkg.apis.policy.v1alpha1
)
)

informergen_external_apis_csv=$(IFS=,; echo "${informergen_external_apis[*]}")

${informergen} \
  --output-base "${KUBE_ROOT}/vendor" \
  --output-package "k8s.io/client-go/informers" \
  --single-directory \
  --input-dirs "${informergen_external_apis_csv}" \
  --versioned-clientset-package k8s.io/client-go/kubernetes \
  --listers-package k8s.io/client-go/listers \
  --go-header-file ${KUBE_ROOT}/hack/boilerplate/boilerplate.generatego.txt \
  "$@"

# You may add additional calls of code generators like set-gen above.

# call generation on sub-project for now
CODEGEN_PKG=./vendor/k8s.io/code-generator vendor/k8s.io/code-generator/hack/update-codegen.sh
CODEGEN_PKG=./vendor/k8s.io/code-generator vendor/k8s.io/kube-aggregator/hack/update-codegen.sh
CODEGEN_PKG=./vendor/k8s.io/code-generator vendor/k8s.io/sample-apiserver/hack/update-codegen.sh
CODEGEN_PKG=./vendor/k8s.io/code-generator vendor/k8s.io/sample-controller/hack/update-codegen.sh
CODEGEN_PKG=./vendor/k8s.io/code-generator vendor/k8s.io/apiextensions-apiserver/hack/update-codegen.sh
CODEGEN_PKG=./vendor/k8s.io/code-generator vendor/k8s.io/metrics/hack/update-codegen.sh
CODEGEN_PKG=./vendor/k8s.io/code-generator vendor/k8s.io/apiextensions-apiserver/examples/client-go/hack/update-codegen.sh
CODEGEN_PKG=./vendor/k8s.io/code-generator vendor/k8s.io/csi-api/hack/update-codegen.sh
