#!/usr/bin/env bash
# Copyright 2023 The Kubernetes Authors.
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

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"
cd "${KUBE_ROOT}"

# Files larger than 1MB need to be allowed explicitly.
maxsize=$((1 * 1024 * 1024))

# Sorted list of those exceptions.
allowlist=(
    staging/src/k8s.io/kubectl/images/kubectl-logo-full.png
)


# Files larger than 1MB get reported and verification fails, unless the file is
# allowed to be larger. Any output or a non-zero exit status indicate a
# failure.
largefiles () {
    # --eol adds information that allows detecting binary files:
    #    i/-text w/-text attr/text=auto eol=lf 	test/images/sample-device-plugin/sampledeviceplugin
    #
    # shellcheck disable=SC2034 # Some variables unused and only present to capture the output field.
    git ls-files -cm --exclude-standard ':!:vendor/*' --eol | while read -r index tree attr eol file; do
        case "$tree" in
            w/-text)
                # Only binary files have a size limit.
                size="$(stat --printf=%s "$file")"
                if [ "${size}" -gt "$maxsize" ] &&
                       ! kube::util::array_contains "$file" "${allowlist[@]}"; then
                    echo    "$file is too large ($size bytes)"
                fi
                ;;
            w/|w/lf|w/crlf|w/mixed|w/none)
                # Other files are okay.
                ;;
            *)
                echo "unexpected 'git ls-files --eol' output for $file: $tree"
                ;;
        esac
    done
}

if ! result="$(largefiles 2>&1)" || [ "${result}" != "" ]; then
    # Help "make verify" and Prow identify the failure message through the "ERROR" prefix.
    sed -e 's/^/ERROR: /' <<EOF
$result

Large binary files should not be committed to git. Exceptions need to be listed in
${BASH_SOURCE[0]}.
EOF
    exit 1
fi
