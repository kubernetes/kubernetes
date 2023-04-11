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
Usage: $(basename "$0") <generators> <output-package> <internal-apis-package> <extensiona-apis-package> <groups-versions> ...

  <generators>        the generators comma separated to run (applyconfiguration,client,conversion,deepcopy,defaulter,informer,lister,openapi).
  <output-package>    the output package name (e.g. github.com/example/project/pkg/generated).
  <int-apis-package>  the internal types dir (e.g. github.com/example/project/pkg/apis) or "" if none.
  <ext-apis-package>  the external types dir (e.g. github.com/example/project/pkg/apis or githubcom/example/apis).
  <groups-versions>   the groups and their versions in the format "groupA:v1,v2 groupB:v1 groupC:v2", relative
                      to <api-package>.
  ...                 arbitrary flags passed to all generator binaries.

Example:
  $(basename "$0") \
      deepcopy,defaulter,conversion \
      github.com/example/project/pkg/client \
      github.com/example/project/pkg/apis \
      github.com/example/project/apis \
      "foo:v1 bar:v1alpha1,v1beta1"
EOF
  exit 0
fi

GENS="$1"
OUTPUT_PKG="$2"
INT_APIS_PKG="$3"
EXT_APIS_PKG="$4"
GROUPS_WITH_VERSIONS="$5"
shift 5

if [ "${GENS}" = "all" ] || grep -qw "all" <<<"${GENS}"; then
    ALL="client,conversion,deepcopy,defaulter,informer,lister,openapi"
    echo "WARNING: Specifying \"all\" as a generator is deprecated."
    echo "WARNING: Please list the specific generators needed."
    echo "WARNING: \"all\" is now an alias for \"${ALL}\"; new code generators WILL NOT be added to this set"
    echo
    GENS="${ALL}"
fi

(
  # To support running this script from anywhere, first cd into this directory,
  # and then install with forced module mode on and fully qualified name.
  cd "$(dirname "${0}")"
  BINS=(
      applyconfiguration-gen
      client-gen
      conversion-gen
      deepcopy-gen
      defaulter-gen
      informer-gen
      lister-gen
      openapi-gen
  )
  # Compile all the tools at once - it's slightly faster but also just simpler.
  # shellcheck disable=2046 # printf word-splitting is intentional
  GO111MODULE=on go install $(printf "k8s.io/code-generator/cmd/%s " "${BINS[@]}")
)

# Go installs the above commands to get installed in $GOBIN if defined, and $GOPATH/bin otherwise:
GOBIN="$(go env GOBIN)"
gobin="${GOBIN:-$(go env GOPATH)/bin}"

function codegen::join() { local IFS="$1"; shift; echo "$*"; }

# enumerate group versions
ALL_FQ_APIS=() # e.g. k8s.io/kubernetes/pkg/apis/apps k8s.io/api/apps/v1
EXT_FQ_APIS=() # e.g. k8s.io/api/apps/v1
for GVs in ${GROUPS_WITH_VERSIONS}; do
  IFS=: read -r G Vs <<<"${GVs}"

  if [ -n "${INT_APIS_PKG}" ]; then
    ALL_FQ_APIS+=("${INT_APIS_PKG}/${G}")
  fi

  # enumerate versions
  for V in ${Vs//,/ }; do
    ALL_FQ_APIS+=("${EXT_APIS_PKG}/${G}/${V}")
    EXT_FQ_APIS+=("${EXT_APIS_PKG}/${G}/${V}")
  done
done

if grep -qw "deepcopy" <<<"${GENS}"; then
  echo "Generating deepcopy funcs"
  "${gobin}/deepcopy-gen" \
      --input-dirs "$(codegen::join , "${ALL_FQ_APIS[@]}")" \
      -O zz_generated.deepcopy \
      "$@"
fi

if grep -qw "defaulter" <<<"${GENS}"; then
  echo "Generating defaulters"
  "${gobin}/defaulter-gen"  \
      --input-dirs "$(codegen::join , "${EXT_FQ_APIS[@]}")" \
      -O zz_generated.defaults \
      "$@"
fi

if grep -qw "conversion" <<<"${GENS}"; then
  echo "Generating conversions"
  "${gobin}/conversion-gen" \
      --input-dirs "$(codegen::join , "${ALL_FQ_APIS[@]}")" \
      -O zz_generated.conversion \
      "$@"
fi

if grep -qw "applyconfiguration" <<<"${GENS}"; then
  echo "Generating apply configuration for ${GROUPS_WITH_VERSIONS} at ${OUTPUT_PKG}/${APPLYCONFIGURATION_PKG_NAME:-applyconfiguration}"
  APPLY_CONFIGURATION_PACKAGE="${OUTPUT_PKG}/${APPLYCONFIGURATION_PKG_NAME:-applyconfiguration}"
  "${gobin}/applyconfiguration-gen" \
      --input-dirs "$(codegen::join , "${EXT_FQ_APIS[@]}")" \
      --output-package "${APPLY_CONFIGURATION_PACKAGE}" \
      "$@"
fi

if grep -qw "client" <<<"${GENS}"; then
  echo "Generating clientset for ${GROUPS_WITH_VERSIONS} at ${OUTPUT_PKG}/${CLIENTSET_PKG_NAME:-clientset}"
  "${gobin}/client-gen" \
      --clientset-name "${CLIENTSET_NAME_VERSIONED:-versioned}" \
      --input-base "" \
      --input "$(codegen::join , "${EXT_FQ_APIS[@]}")" \
      --output-package "${OUTPUT_PKG}/${CLIENTSET_PKG_NAME:-clientset}" \
      --apply-configuration-package "${APPLY_CONFIGURATION_PACKAGE:-}" \
      "$@"
fi

if grep -qw "lister" <<<"${GENS}"; then
  echo "Generating listers for ${GROUPS_WITH_VERSIONS} at ${OUTPUT_PKG}/listers"
  "${gobin}/lister-gen" \
      --input-dirs "$(codegen::join , "${EXT_FQ_APIS[@]}")" \
      --output-package "${OUTPUT_PKG}/listers" \
      "$@"
fi

if grep -qw "informer" <<<"${GENS}"; then
  echo "Generating informers for ${GROUPS_WITH_VERSIONS} at ${OUTPUT_PKG}/informers"
  "${gobin}/informer-gen" \
      --input-dirs "$(codegen::join , "${EXT_FQ_APIS[@]}")" \
      --versioned-clientset-package "${OUTPUT_PKG}/${CLIENTSET_PKG_NAME:-clientset}/${CLIENTSET_NAME_VERSIONED:-versioned}" \
      --listers-package "${OUTPUT_PKG}/listers" \
      --output-package "${OUTPUT_PKG}/informers" \
      "$@"
fi

if grep -qw "openapi" <<<"${GENS}"; then
  echo "Generating OpenAPI definitions for ${GROUPS_WITH_VERSIONS} at ${OUTPUT_PKG}/openapi"
  declare -a OPENAPI_EXTRA_PACKAGES
  "${gobin}/openapi-gen" \
      --input-dirs "$(codegen::join , "${EXT_FQ_APIS[@]}" "${OPENAPI_EXTRA_PACKAGES[@]+"${OPENAPI_EXTRA_PACKAGES[@]}"}")" \
      --input-dirs "k8s.io/apimachinery/pkg/apis/meta/v1,k8s.io/apimachinery/pkg/runtime,k8s.io/apimachinery/pkg/version" \
      --output-package "${OUTPUT_PKG}/openapi" \
      -O zz_generated.openapi \
      "$@"
fi
