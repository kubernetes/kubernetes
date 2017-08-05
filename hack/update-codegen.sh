#!/bin/bash

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
  vendor/k8s.io/kube-gen/cmd/client-gen
  vendor/k8s.io/kube-gen/cmd/lister-gen
  vendor/k8s.io/kube-gen/cmd/informer-gen
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
done
# delimit by commas for the command
GV_DIRS_CSV=$(IFS=',';echo "${GV_DIRS[*]// /,}";IFS=$)

# This can be called with one flag, --verify-only, so it works for both the
# update- and verify- scripts.
${clientgen} "$@"
${clientgen} --output-base "${KUBE_ROOT}/vendor" --clientset-path="k8s.io/client-go" --clientset-name="kubernetes" --input-base="k8s.io/kubernetes/vendor/k8s.io/api" --input="${GV_DIRS_CSV}" "$@"
# Clientgen for federation clientset.
${clientgen} --clientset-name=federation_clientset --clientset-path=k8s.io/kubernetes/federation/client/clientset_generated --input-base="k8s.io/kubernetes/vendor/k8s.io/api" --input="../../../federation/apis/federation/v1beta1","core/v1","extensions/v1beta1","batch/v1","autoscaling/v1" --included-types-overrides="core/v1/Service,core/v1/Namespace,extensions/v1beta1/ReplicaSet,core/v1/Secret,extensions/v1beta1/Ingress,extensions/v1beta1/Deployment,extensions/v1beta1/DaemonSet,core/v1/ConfigMap,core/v1/Event,batch/v1/Job,autoscaling/v1/HorizontalPodAutoscaler"   "$@"

listergen_internal_apis=(
pkg/api
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
  # because client-gen doesn't do policy/v1alpha1, we have to skip it too
  find k8s.io/api -name types.go | xargs -n1 dirname | sort | grep -v pkg.apis.policy.v1alpha1
)
)
listergen_external_apis_csv=$(IFS=,; echo "${listergen_external_apis[*]}")
${listergen} --output-base "${KUBE_ROOT}/vendor" --output-package "k8s.io/client-go/listers" --input-dirs "${listergen_external_apis_csv}" "$@"

informergen_internal_apis=(
pkg/api
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
  "$@"

# You may add additional calls of code generators like set-gen above.

# call generation on sub-project for now
KUBEGEN_PKG=./vendor/k8s.io/kube-gen vendor/k8s.io/kube-gen/hack/update-codegen.sh
KUBEGEN_PKG=./vendor/k8s.io/kube-gen vendor/k8s.io/kube-aggregator/hack/update-codegen.sh
KUBEGEN_PKG=./vendor/k8s.io/kube-gen vendor/k8s.io/sample-apiserver/hack/update-codegen.sh
KUBEGEN_PKG=./vendor/k8s.io/kube-gen vendor/k8s.io/apiextensions-apiserver/hack/update-codegen.sh
KUBEGEN_PKG=./vendor/k8s.io/kube-gen vendor/k8s.io/metrics/hack/update-codegen.sh
