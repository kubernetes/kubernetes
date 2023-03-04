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

# generate-internal-groups is a back-compatible wrapper around generate-groups.sh

if [ "$#" -lt 5 ] || [ "${1}" == "--help" ]; then
  cat <<EOF
Usage: $(basename "$0") <generators> <output-package> <int-apis-package> <apis-package> <groups-versions> ...

  <generators>        the generators comma separated to run (deepcopy,defaulter,conversion,client,lister,informer,openapi) or "all".
  <output-package>    the output package name (e.g. github.com/example/project/pkg/generated).
  <int-apis-package>  Deprecated but retained for compatibility (has no effect).
  <apis-package>      the external types dir (e.g. github.com/example/project/pkg/apis or githubcom/example/apis).
  <groups-versions>   the groups and their versions in the format "groupA:v1,v2 groupB:v1 groupC:v2", relative
                      to <api-package>.
  ...                 arbitrary flags passed to all generator binaries.

EOF
  exit 0
fi

GENS="$1"
OUTPUT_PKG="$2"
APIS_PKG="$3"
# $4 is deprecated
GROUPS_WITH_VERSIONS="$5"
shift 5

if [ "${GENS}" = "all" ]; then
    # Don't pass "all" thru to generate-groups because it has a different meaning there.
    GENS="client,conversion,deepcopy,defaulter,informer,lister,openapi"
fi

echo "WARNING: generate-internal-groups.sh is deprecated: use generate-groups.sh instead"
exec "$(dirname "${BASH_SOURCE[0]}")/generate-groups.sh" "${GENS}" "${OUTPUT_PKG}" "${APIS_PKG}" "${GROUPS_WITH_VERSIONS}" "$@"
