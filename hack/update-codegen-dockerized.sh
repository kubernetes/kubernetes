#!/usr/bin/env bash

# Copyright 2018 The Kubernetes Authors.
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
  vendor/k8s.io/code-generator/cmd/go-to-protobuf
  vendor/k8s.io/code-generator/cmd/go-to-protobuf/protoc-gen-gogo
  vendor/k8s.io/code-generator/cmd/lister-gen
  vendor/k8s.io/code-generator/cmd/informer-gen
)
make -C "${KUBE_ROOT}" WHAT="${BUILD_TARGETS[*]}"

clientgen=$(kube::util::find-binary "client-gen")
gotoprotobuf=$(kube::util::find-binary "go-to-protobuf")
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

listergen_internal_apis=(
$(
  cd ${KUBE_ROOT}
  find pkg/apis -maxdepth 2 -name types.go | xargs -n1 dirname | sort
)
)
listergen_internal_apis=(${listergen_internal_apis[@]/#/k8s.io/kubernetes/})
listergen_internal_apis_csv=$(IFS=,; echo "${listergen_internal_apis[*]}")
${listergen} --input-dirs "${listergen_internal_apis_csv}" "$@"

listergen_external_apis=(
$(
  cd ${KUBE_ROOT}/staging/src
  find k8s.io/api -name types.go | xargs -n1 dirname | sort
)
)
listergen_external_apis_csv=$(IFS=,; echo "${listergen_external_apis[*]}")
${listergen} --output-base "${KUBE_ROOT}/vendor" --output-package "k8s.io/client-go/listers" --input-dirs "${listergen_external_apis_csv}" --go-header-file ${KUBE_ROOT}/hack/boilerplate/boilerplate.generatego.txt "$@"

informergen_internal_apis=(
$(
  cd ${KUBE_ROOT}
  find pkg/apis -maxdepth 2 -name types.go | xargs -n1 dirname | sort
)
)
informergen_internal_apis=(${informergen_internal_apis[@]/#/k8s.io/kubernetes/})
informergen_internal_apis_csv=$(IFS=,; echo "${informergen_internal_apis[*]}")
${informergen} \
  --input-dirs "${informergen_internal_apis_csv}" \
  --internal-clientset-package k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset \
  --listers-package k8s.io/kubernetes/pkg/client/listers \
  --go-header-file ${KUBE_ROOT}/hack/boilerplate/boilerplate.generatego.txt \
  "$@"

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

if [[ -z "$(which protoc)" || "$(protoc --version)" != "libprotoc 3."* ]]; then
  echo "Generating protobuf requires protoc 3.0.0-beta1 or newer. Please download and"
  echo "install the platform appropriate Protobuf package for your OS: "
  echo
  echo "  https://github.com/google/protobuf/releases"
  echo
  echo "WARNING: Protobuf changes are not being validated"
  exit 1
fi

PROTO_PACKAGES=(
  k8s.io/api/core/v1
  k8s.io/api/policy/v1beta1
  k8s.io/api/extensions/v1beta1
  k8s.io/api/autoscaling/v1
  k8s.io/api/authorization/v1
  k8s.io/api/autoscaling/v2beta1
  k8s.io/api/authorization/v1beta1
  k8s.io/api/batch/v1
  k8s.io/api/batch/v1beta1
  k8s.io/api/batch/v2alpha1
  k8s.io/api/apps/v1beta1
  k8s.io/api/apps/v1beta2
  k8s.io/api/apps/v1
  k8s.io/api/authentication/v1
  k8s.io/api/authentication/v1beta1
  k8s.io/api/events/v1beta1
  k8s.io/api/rbac/v1alpha1
  k8s.io/api/rbac/v1beta1
  k8s.io/api/rbac/v1
  k8s.io/api/certificates/v1beta1
  k8s.io/api/coordination/v1beta1
  k8s.io/api/imagepolicy/v1alpha1
  k8s.io/api/scheduling/v1alpha1
  k8s.io/api/scheduling/v1beta1
  k8s.io/api/settings/v1alpha1
  k8s.io/api/storage/v1alpha1
  k8s.io/api/storage/v1beta1
  k8s.io/api/storage/v1
  k8s.io/api/admissionregistration/v1alpha1
  k8s.io/api/admissionregistration/v1beta1
  k8s.io/api/admission/v1beta1
  k8s.io/api/networking/v1
  k8s.io/apiserver/pkg/apis/audit/v1alpha1
  k8s.io/apiserver/pkg/apis/audit/v1beta1
  k8s.io/apiserver/pkg/apis/audit/v1
  k8s.io/apiserver/pkg/apis/example/v1
  k8s.io/apiserver/pkg/apis/example2/v1
  k8s.io/kube-aggregator/pkg/apis/apiregistration/v1beta1
  k8s.io/kube-aggregator/pkg/apis/apiregistration/v1
)

# requires the 'proto' tag to build (will remove when ready)
# searches for the protoc-gen-gogo extension in the output directory
# satisfies import of github.com/gogo/protobuf/gogoproto/gogo.proto and the
# core Google protobuf types
PATH="${KUBE_ROOT}/_output/bin:${PATH}" \
    ${gotoprotobuf} \
        --proto-import "${KUBE_ROOT}/vendor" \
        --proto-import "${KUBE_ROOT}/third_party/protobuf" \
        --packages=$(IFS=, ; echo "${PROTO_PACKAGES[*]}") \
        --go-header-file=${KUBE_ROOT}/hack/boilerplate/boilerplate.generatego.txt \
        "$@"

# You may add additional calls of code generators like set-gen above.

# call generation on sub-project for now
SUB_PROJECT_OP="update-codegen.sh"
if [[ "$@" =~ "--verify-only" ]]; then
    SUB_PROJECT_OP="verify-codegen.sh"
fi

CODEGEN_PKG=./vendor/k8s.io/code-generator vendor/k8s.io/code-generator/hack/$SUB_PROJECT_OP
CODEGEN_PKG=./vendor/k8s.io/code-generator vendor/k8s.io/kube-aggregator/hack/$SUB_PROJECT_OP
CODEGEN_PKG=./vendor/k8s.io/code-generator vendor/k8s.io/sample-apiserver/hack/$SUB_PROJECT_OP
CODEGEN_PKG=./vendor/k8s.io/code-generator vendor/k8s.io/sample-controller/hack/$SUB_PROJECT_OP
CODEGEN_PKG=./vendor/k8s.io/code-generator vendor/k8s.io/apiextensions-apiserver/hack/$SUB_PROJECT_OP
CODEGEN_PKG=./vendor/k8s.io/code-generator vendor/k8s.io/metrics/hack/$SUB_PROJECT_OP
CODEGEN_PKG=./vendor/k8s.io/code-generator vendor/k8s.io/apiextensions-apiserver/examples/client-go/hack/$SUB_PROJECT_OP
CODEGEN_PKG=./vendor/k8s.io/code-generator vendor/k8s.io/csi-api/hack/$SUB_PROJECT_OP

