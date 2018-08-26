#!/usr/bin/env bash

# Copyright 2017 The Kubernetes Authors.
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

# generate-internal-groups generates everything for a project with internal types, e.g. an
# user-provided API server based on k8s.io/apiserver.

if [ "$#" -lt 5 ] || [ "${1}" == "--help" ]; then
  cat <<EOF
Usage: $(basename $0) <generators> <output-package> <internal-apis-package> <extensiona-apis-package> <groups-versions> ...

  <generators>        the generators comma separated to run (deepcopy,defaulter,conversion,client,lister,informer) or "all".
  <output-package>    the output package name (e.g. github.com/example/project/pkg/generated).
  <int-apis-package>  the internal types dir (e.g. github.com/example/project/pkg/apis).
  <ext-apis-package>  the external types dir (e.g. github.com/example/project/pkg/apis or githubcom/example/apis).
  <groups-versions>   the groups and their versions in the format "groupA:v1,v2 groupB:v1 groupC:v2", relative
                      to <api-package>.
  ...                 arbitrary flags passed to all generator binaries.

Examples:
  $(basename $0) all                           github.com/example/project/pkg/client github.com/example/project/pkg/apis github.com/example/project/pkg/apis "foo:v1 bar:v1alpha1,v1beta1"
  $(basename $0) deepcopy,defaulter,conversion github.com/example/project/pkg/client github.com/example/project/pkg/apis github.com/example/project/apis     "foo:v1 bar:v1alpha1,v1beta1"
EOF
  exit 0
fi

GENS="$1"
OUTPUT_PKG="$2"
INT_APIS_PKG="$3"
EXT_APIS_PKG="$4"
GROUPS_WITH_VERSIONS="$5"
shift 5

go install ./$(dirname "${0}")/cmd/{defaulter-gen,conversion-gen,client-gen,lister-gen,informer-gen,deepcopy-gen,go-to-protobuf,go-to-protobuf/protoc-gen-gogo}
function codegen::join() { local IFS="$1"; shift; echo "$*"; }

# enumerate group versions
ALL_FQ_APIS=() # e.g. k8s.io/kubernetes/pkg/apis/apps k8s.io/api/apps/v1
INT_FQ_APIS=() # e.g. k8s.io/kubernetes/pkg/apis/apps
EXT_FQ_APIS=() # e.g. k8s.io/api/apps/v1
for GVs in ${GROUPS_WITH_VERSIONS}; do
  IFS=: read G Vs <<<"${GVs}"

  if [ -n "${INT_APIS_PKG}" ]; then
    ALL_FQ_APIS+=("${INT_APIS_PKG}/${G}")
    INT_FQ_APIS+=("${INT_APIS_PKG}/${G}")
  fi

  # enumerate versions
  for V in ${Vs//,/ }; do
    ALL_FQ_APIS+=("${EXT_APIS_PKG}/${G}/${V}")
    EXT_FQ_APIS+=("${EXT_APIS_PKG}/${G}/${V}")
  done
done

if [ "${GENS}" = "all" ] || grep -qw "deepcopy" <<<"${GENS}"; then
  echo "Generating deepcopy funcs"
  ${GOPATH}/bin/deepcopy-gen --input-dirs $(codegen::join , "${ALL_FQ_APIS[@]}") -O zz_generated.deepcopy --bounding-dirs ${INT_APIS_PKG},${EXT_APIS_PKG} "$@"
fi

if [ "${GENS}" = "all" ] || grep -qw "defaulter" <<<"${GENS}"; then
  echo "Generating defaulters"
  ${GOPATH}/bin/defaulter-gen  --input-dirs $(codegen::join , "${EXT_FQ_APIS[@]}") -O zz_generated.defaults "$@"
fi

if [ "${GENS}" = "all" ] || grep -qw "conversion" <<<"${GENS}"; then
  echo "Generating conversions"
  ${GOPATH}/bin/conversion-gen --input-dirs $(codegen::join , "${ALL_FQ_APIS[@]}") -O zz_generated.conversion "$@"
fi

if [ "${GENS}" = "all" ] || grep -qw "client" <<<"${GENS}"; then
  echo "Generating clientset for ${GROUPS_WITH_VERSIONS} at ${OUTPUT_PKG}/clientset"
  if [ -n "${INT_APIS_PKG}" ]; then
    ${GOPATH}/bin/client-gen --clientset-name ${CLIENTSET_NAME_INTERNAL:-internalversion} --input-base "" --input $(codegen::join , $(printf '%s/ ' "${INT_FQ_APIS[@]}")) --output-package ${OUTPUT_PKG}/clientset "$@"
  fi
  ${GOPATH}/bin/client-gen --clientset-name ${CLIENTSET_NAME_VERSIONED:-versioned} --input-base "" --input $(codegen::join , "${EXT_FQ_APIS[@]}") --output-package ${OUTPUT_PKG}/clientset "$@"
fi

if [ "${GENS}" = "all" ] || grep -qw "lister" <<<"${GENS}"; then
  echo "Generating listers for ${GROUPS_WITH_VERSIONS} at ${OUTPUT_PKG}/listers"
  ${GOPATH}/bin/lister-gen --input-dirs $(codegen::join , "${ALL_FQ_APIS[@]}") --output-package ${OUTPUT_PKG}/listers "$@"
fi

if [ "${GENS}" = "all" ] || grep -qw "informer" <<<"${GENS}"; then
  echo "Generating informers for ${GROUPS_WITH_VERSIONS} at ${OUTPUT_PKG}/informers"
  ${GOPATH}/bin/informer-gen \
           --input-dirs $(codegen::join , "${ALL_FQ_APIS[@]}") \
           --versioned-clientset-package ${OUTPUT_PKG}/clientset/${CLIENTSET_NAME_VERSIONED:-versioned} \
           --internal-clientset-package ${OUTPUT_PKG}/clientset/${CLIENTSET_NAME_INTERNAL:-internalversion} \
           --listers-package ${OUTPUT_PKG}/listers \
           --output-package ${OUTPUT_PKG}/informers \
           "$@"
fi

if [ "${GENS}" = "all" ] || grep -qw "protobuf" <<<"${GENS}"; then
  PROTO_PACKAGES=(
    k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1
    k8s.io/metrics/pkg/apis/metrics/v1alpha1
    k8s.io/metrics/pkg/apis/metrics/v1beta1
    k8s.io/metrics/pkg/apis/custom_metrics/v1beta1
    k8s.io/metrics/pkg/apis/external_metrics/v1beta1
    k8s.io/sample-apiserver/pkg/apis/wardle/v1alpha1
    k8s.io/sample-apiserver/pkg/apis/wardle/v1beta1
  )

  if [[ " ${PROTO_PACKAGES[@]} " =~ " ${EXT_FQ_APIS[@]} " ]]; then
    echo "Generating protobuf for ${GROUPS_WITH_VERSIONS}"

    if [[ -z "$(which protoc)" || "$(protoc --version)" != "libprotoc 3."* ]]; then
      echo "Generating protobuf requires protoc 3.0.0-beta1 or newer. Please download and"
      echo "install the platform appropriate Protobuf package for your OS: "
      echo
      echo "  https://github.com/google/protobuf/releases"
      echo
      echo "WARNING: Protobuf changes are not being validated"
      exit 1
    fi

    # requires the 'proto' tag to build (will remove when ready)
    # searches for the protoc-gen-gogo extension in the output directory
    # satisfies import of github.com/gogo/protobuf/gogoproto/gogo.proto and the
    # core Google protobuf types
    PATH="./$(dirname "${0}")/_output/bin:${PATH}" \
      ${GOPATH}/bin/go-to-protobuf \
      --proto-import "./vendor" \
      --packages $(codegen::join , "${EXT_FQ_APIS[@]}") \
      "$@" \
      --output-base "$GOPATH/src"
  fi
fi

