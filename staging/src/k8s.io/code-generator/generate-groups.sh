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

# generate-groups generates everything for a project with external types only, e.g. a project based
# on CustomResourceDefinitions.

if [ "$#" -lt 4 ] || [ "${1}" == "--help" ]; then
  cat <<EOF
Usage: $(basename "$0") <generators> <output-package> <apis-package> <groups-versions> ...

  <generators>        the generators comma separated to run (deepcopy,defaulter,applyconfiguration,client,lister,informer).
  <output-package>    the output package name (e.g. github.com/example/project/pkg/generated).
  <apis-package>      the external types dir (e.g. github.com/example/api or github.com/example/project/pkg/apis).
  <groups-versions>   the groups and their versions in the format "groupA:v1,v2 groupB:v1 groupC:v2", relative
                      to <api-package>.
  ...                 arbitrary flags passed to all generator binaries.


Example:
  $(basename "$0") \
      deepcopy,client \
      github.com/example/project/pkg/client \
      github.com/example/project/pkg/apis \
      "foo:v1 bar:v1alpha1,v1beta1"
EOF
  exit 0
fi

GENS="$1"
OUTPUT_PKG="$2"
APIS_PKG="$3"
GROUPS_WITH_VERSIONS="$4"
shift 4

echo "WARNING: $(basename "$0") is deprecated."
echo "WARNING: Please use k8s.io/code-generator/kube_codegen.sh instead."
echo

if [ "${GENS}" = "all" ] || grep -qw "all" <<<"${GENS}"; then
    ALL="applyconfiguration,client,deepcopy,informer,lister"
    echo "WARNING: Specifying \"all\" as a generator is deprecated."
    echo "WARNING: Please list the specific generators needed."
    echo "WARNING: \"all\" is now an alias for \"${ALL}\"; new code generators WILL NOT be added to this set"
    echo
    GENS="${ALL}"
fi

INT_APIS_PKG=""
exec "$(dirname "${BASH_SOURCE[0]}")/generate-internal-groups.sh" "${GENS}" "${OUTPUT_PKG}" "${INT_APIS_PKG}" "${APIS_PKG}" "${GROUPS_WITH_VERSIONS}" "$@"
