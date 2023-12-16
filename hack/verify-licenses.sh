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

# Usage: `hack/verify-licenses.sh`.


set -o errexit
set -o nounset
set -o pipefail


KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"
source "${KUBE_ROOT}/hack/lib/util.sh"

# This sets up the environment, like GOCACHE, which keeps the worktree cleaner.
kube::golang::setup_env
kube::util::ensure-temp-dir

# Creating a new repository tree 
# Deleting vendor directory to make go-licenses fetch license URLs from go-packages source repository
git worktree add -f "${KUBE_TEMP}"/tmp_test_licenses/kubernetes HEAD >/dev/null 2>&1 || true
cd "${KUBE_TEMP}"/tmp_test_licenses/kubernetes && rm -rf vendor

# Ensure that we find the binaries we build before anything else.
export GOBIN="${KUBE_OUTPUT_BINPATH}"
PATH="${GOBIN}:${PATH}"

# Explicitly opt into go modules, even though we're inside a GOPATH directory
export GO111MODULE=on

function http_code() {
    curl -I -s -o /dev/null -w "%{http_code}" "$1"
}

allowed_licenses=()
packages_flagged=()
packages_url_missing=()
exit_code=0

# Install go-licenses
echo '[INFO] Installing go-licenses...'
go install github.com/google/go-licenses@latest

# Fetching CNCF Approved List Of Licenses
# Refer: https://github.com/cncf/foundation/blob/main/allowed-third-party-license-policy.md
curl -s 'https://spdx.org/licenses/licenses.json' -o "${KUBE_TEMP}"/licenses.json

echo '[INFO] Fetching current list of CNCF approved licenses...'
while read -r L; do
    allowed_licenses+=("${L}")
done < <(jq -r '.licenses[] | select(.isDeprecatedLicenseId==false) .licenseId' "${KUBE_TEMP}"/licenses.json)

# Scanning go-packages under the project & verifying against the CNCF approved list of licenses
echo '[INFO] Starting license scan on go-packages...'
go-licenses report ./... >> "${KUBE_TEMP}"/licenses.csv

echo -e 'PACKAGE_NAME  LICENSE_NAME  LICENSE_URL\n' >> "${KUBE_TEMP}"/approved_licenses.dump
while IFS=, read -r GO_PACKAGE LICENSE_URL LICENSE_NAME; do
    if ! printf -- "%s\n" "${allowed_licenses[@]}" | grep -q "^${LICENSE_NAME}$"; then
        echo "${GO_PACKAGE}  ${LICENSE_NAME}  ${LICENSE_URL}" >> "${KUBE_TEMP}"/notapproved_licenses.dump
        packages_flagged+=("${GO_PACKAGE}")
        continue
    fi

    if [[ "${LICENSE_URL}" == 'Unknown' ]]; then
        if  [[ "${GO_PACKAGE}" != k8s.io/* ]]; then
            echo "${GO_PACKAGE}  ${LICENSE_NAME}  ${LICENSE_URL}" >> "${KUBE_TEMP}"/approved_licenses_with_missing_urls.dump
            packages_url_missing+=("${GO_PACKAGE}")
        else
            LICENSE_URL='https://github.com/kubernetes/kubernetes/blob/master/LICENSE'
            echo "${GO_PACKAGE}  ${LICENSE_NAME}  ${LICENSE_URL}" >> "${KUBE_TEMP}"/approved_licenses.dump
        fi
        continue
    fi

    if [[ "$(http_code "${LICENSE_URL}")" != 404 ]]; then
        echo "${GO_PACKAGE}  ${LICENSE_NAME}  ${LICENSE_URL}" >> "${KUBE_TEMP}"/approved_licenses.dump
        continue
    fi

    # The URL 404'ed.  Try parent-paths.

    #echo -e "DBG: err 404 ${LICENSE_URL}"
    dir="$(dirname "${LICENSE_URL}")"
    file="$(basename "${LICENSE_URL}")"

    while [[ "${dir}" != "." ]]; do
        dir="$(dirname "${dir}")"
        #echo "DBG:     try ${dir}/${file}"
        if [[ "$(http_code "${dir}/${file}")" != 404 ]]; then
            #echo "DBG:         it worked"
            echo "${GO_PACKAGE}  ${LICENSE_NAME}  ${dir}/${file}" >> "${KUBE_TEMP}"/approved_licenses.dump
            break
        fi
        #echo "DBG:         still 404"
    done
    if [[ "${dir}" == "." ]];then
        #echo "DBG:     failed to find a license"
        packages_url_missing+=("${GO_PACKAGE}")
        echo "${GO_PACKAGE}  ${LICENSE_NAME}  ${LICENSE_URL}" >> "${KUBE_TEMP}"/approved_licenses_with_missing_urls.dump
    fi
done < "${KUBE_TEMP}"/licenses.csv
awk '{ printf "%-100s : %-20s : %s\n", $1, $2, $3 }' "${KUBE_TEMP}"/approved_licenses.dump


if [[ ${#packages_url_missing[@]} -gt 0 ]]; then
    echo -e '\n[ERROR] The following go-packages in the project have unknown or unreachable license URL:'
    awk '{ printf "%-100s :  %-20s : %s\n", $1, $2, $3 }' "${KUBE_TEMP}"/approved_licenses_with_missing_urls.dump
    exit_code=1
fi


if [[ ${#packages_flagged[@]} -gt 0 ]]; then
    kube::log::error "[ERROR] The following go-packages in the project are using non-CNCF approved licenses. Please refer to the CNCF's approved licence list for further information: https://github.com/cncf/foundation/blob/main/allowed-third-party-license-policy.md"
    awk '{ printf "%-100s :  %-20s : %s\n", $1, $2, $3 }' "${KUBE_TEMP}"/notapproved_licenses.dump
    exit_code=1
elif [[ "${exit_code}" -eq 1 ]]; then
    kube::log::status "[ERROR] Project is using go-packages with unknown or unreachable license URLs. Please refer to the CNCF's approved licence list for further information: https://github.com/cncf/foundation/blob/main/allowed-third-party-license-policy.md"
else
    kube::log::status "[SUCCESS] Scan complete! All go-packages under the project are using current CNCF approved licenses!"
fi

exit "${exit_code}"
