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
  cmd/libs/go2idl/client-gen
  cmd/libs/go2idl/conversion-gen
  cmd/libs/go2idl/deepcopy-gen
  cmd/libs/go2idl/set-gen
)
"${KUBE_ROOT}/hack/build-go.sh" ${BUILD_TARGETS[*]}

clientgen=$(kube::util::find-binary "client-gen")
conversiongen=$(kube::util::find-binary "conversion-gen")
deepcopygen=$(kube::util::find-binary "deepcopy-gen")
setgen=$(kube::util::find-binary "set-gen")

# Please do not add any logic to this shell script. Add logic to the go code
# that generates the set-gen program.
#
# This can be called with one flag, --verify-only, so it works for both the
# update- and verify- scripts.
${clientgen} "$@"
${clientgen} -t "$@"
${clientgen} --clientset-name="release_1_4" --input="api/v1,extensions/v1beta1,autoscaling/v1,batch/v1"
# Clientgen for federation clientset.
${clientgen} --clientset-name=federation_internalclientset --clientset-path=k8s.io/kubernetes/federation/client/clientset_generated --input="../../federation/apis/federation/","api/" --included-types-overrides="api/Service"   "$@"
${clientgen} --clientset-name=federation_release_1_4 --clientset-path=k8s.io/kubernetes/federation/client/clientset_generated --input="../../federation/apis/federation/v1beta1","api/v1" --included-types-overrides="api/v1/Service"   "$@"
${setgen} "$@"

# You may add additional calls of code generators like set-gen above.

# Generate a list of all files that have a `+k8s:` comment-tag.  This will be
# used to derive lists of files/dirs for generation tools.
ALL_K8S_TAG_FILES=$(
    grep -l '^// \?+k8s:' $(
        find . \
            -not \( \
                \( \
                    -path ./vendor -o \
                    -path ./_output -o \
                    -path ./.git \
                \) -prune \
            \) \
            -type f -name \*.go \
            | sed 's|^./||'
        )
    )
DEEP_COPY_DIRS=$(
    grep -l '+k8s:deepcopy-gen=' ${ALL_K8S_TAG_FILES} \
        | xargs dirname \
        | sort -u
    )
DEEPCOPY_INPUTS=$(
    for d in ${DEEP_COPY_DIRS}; do
        echo k8s.io/kubernetes/$d
    done | paste -sd,
    )
${deepcopygen} -i ${DEEPCOPY_INPUTS}

CONVERSION_DIRS=$(
    grep '^// *+k8s:conversion-gen=' ${ALL_K8S_TAG_FILES} \
        | cut -f1 -d:                                     \
        | xargs dirname                                   \
        | sort -u                                         \
    )
CONVERSION_INPUTS=$(
     for d in ${CONVERSION_DIRS}; do
         echo k8s.io/kubernetes/$d
     done | paste -sd,
     )
${conversiongen} -i ${CONVERSION_INPUTS}
