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


kube::golang::verify_go_version
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


allowed_licenses=()
packages_flagged=()
packages_url_missing=()
exit_code=0


git remote add licenses https://github.com/kubernetes/kubernetes >/dev/null 2>&1 || true


# Install go-licenses
echo -e '[INFO] Installing go-licenses...'
pushd "${KUBE_TEMP}" >/dev/null
    git clone https://github.com/google/go-licenses.git >/dev/null 2>&1
    cd go-licenses
    go build -o "${GOPATH}/bin"
popd >/dev/null


# Fetching CNCF Approved List Of Licenses
# Refer: https://github.com/cncf/foundation/blob/main/allowed-third-party-license-policy.md
curl -s 'https://spdx.org/licenses/licenses.json' -o "${KUBE_TEMP}"/licenses.json


number_of_licenses=$(jq '.licenses | length' "${KUBE_TEMP}"/licenses.json)
loop_index_length=$(( number_of_licenses - 1 ))


echo -e '[INFO] Fetching current list of CNCF approved licenses...'
for index in $(seq 0 $loop_index_length);
do
	licenseID=$(jq ".licenses[$index] .licenseId" "${KUBE_TEMP}"/licenses.json)
	if [[ $(jq ".licenses[$index] .isDeprecatedLicenseId" "${KUBE_TEMP}"/licenses.json) == false ]]
	then
		allowed_licenses+=("${licenseID}")
        fi	
done


# Scanning go-packages under the project & verifying against the CNCF approved list of licenses
echo -e '[INFO] Starting license scan on go-packages...'
go-licenses csv --git_remote "licenses" ./... >> "${KUBE_TEMP}"/licenses.csv 2>/dev/null


echo -e 'PACKAGE_NAME  LICENSE_NAME  LICENSE_URL\n' >> "${KUBE_TEMP}"/approved_licenses.dump
while IFS=, read -r GO_PACKAGE LICENSE_URL LICENSE_NAME
do
	if [[ " ${allowed_licenses[*]} " == *"${LICENSE_NAME}"* ]];
	then
		if [[ "${LICENSE_URL}" == 'Unknown' ]];
		then
			if  [[ "${GO_PACKAGE}" != k8s.io/* ]];
			then
				echo -e "${GO_PACKAGE}  ${LICENSE_NAME}  ${LICENSE_URL}" >> "${KUBE_TEMP}"/approved_licenses_with_missing_urls.dump
				packages_url_missing+=("${GO_PACKAGE}")
			else
				LICENSE_URL='https://github.com/kubernetes/kubernetes/blob/master/LICENSE'
				echo -e "${GO_PACKAGE}  ${LICENSE_NAME}  ${LICENSE_URL}" >> "${KUBE_TEMP}"/approved_licenses.dump
			fi
		elif curl -Is "${LICENSE_URL}" | head -1 | grep -q 404;
		then
			packages_url_missing+=("${GO_PACKAGE}")
			echo -e "${GO_PACKAGE}  ${LICENSE_NAME}  ${LICENSE_URL}" >> "${KUBE_TEMP}"/approved_licenses_with_missing_urls.dump
		else
			echo -e "${GO_PACKAGE}  ${LICENSE_NAME}  ${LICENSE_URL}" >> "${KUBE_TEMP}"/approved_licenses.dump
		fi
	else
		echo -e "${GO_PACKAGE}  ${LICENSE_NAME}  ${LICENSE_URL}" >> "${KUBE_TEMP}"notapproved_licenses.dump
		packages_flagged+=("${GO_PACKAGE}")
	fi
done < "${KUBE_TEMP}"/licenses.csv
awk '{ printf "%-100s : %-20s : %s\n", $1, $2, $3 }' "${KUBE_TEMP}"/approved_licenses.dump


# cleanup
git remote remove licenses


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
	kube::log::status "[ERROR] Project is using go-packages with unknown or unreachable license URLs. Please refer to the CNCF's approved licence list for further information: https://github.com/cncf/foundation/blob/main/allowed-third-party-license-policy.md "
else
	kube::log::status "[SUCCESS] Scan complete! All go-packages under the project are using current CNCF approved licenses!"
fi

exit ${exit_code}
