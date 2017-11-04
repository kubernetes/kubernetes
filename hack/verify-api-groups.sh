#!/bin/bash

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

# locate all API groups by their packages and versions


set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"
prefix="${KUBE_ROOT%"k8s.io/kubernetes"}"

register_files=()
while IFS= read -d $'\0' -r file ; do
	register_files+=("${file}")
done < <(find "${KUBE_ROOT}"/pkg/apis -name register.go -print0)

# every register file should contain a GroupName.  Gather the different representations.
# 1. group directory name for client gen
# 2. external group versions for init.sh all APIs list
# 3. install packages for inclusion in import_known_versions files
group_dirnames=()
external_group_versions=()
expected_install_packages=()
for register_file in "${register_files[@]}"; do
	package="${register_file#"${prefix}"}"
	package="${package%"/register.go"}"
	group_dirname="${package#"k8s.io/kubernetes/pkg/apis/"}"
	group_dirname="${group_dirname%%"/*"}"
	group_name=""
	if grep -q 'GroupName = "' "${register_file}"; then
		group_name=$(grep -q 'GroupName = "' "${register_file}" | cut -d\" -f2 -)
	else
		echo "${register_file} is missing \"const GroupName =\""
		exit 1
	fi

	# does the dirname doesn't have a slash, then it's the internal package.
	# if does have one, then its an external
	if [[ "${group_dirname#*'/'}" == "${group_dirname}" ]]; then
		group_dirnames+=("${group_dirname}")
		expected_install_packages+=("${package}")
	else
		version=$(echo "${group_dirname}" | cut -d/ -f2 -)
		external_group_versions+=("${group_name}/${version}")
	fi
done


# check to make sure that client gen is getting
# groups_without_codegen is the list of group we EXPECT to not have the client generated for
# them.  This happens for types that aren't served from the API server
groups_without_codegen=(
	"abac"
	"componentconfig"
	"imagepolicy"
	"admission"
)
client_gen_file="${KUBE_ROOT}/vendor/k8s.io/code-generator/cmd/client-gen/main.go"

for group_dirname in "${group_dirnames[@]}"; do
	if ! grep -q "${group_dirname}/" "${client_gen_file}" ; then
		found=0
		for group_without_codegen in "${groups_without_codegen[@]}"; do
			if [[ "${group_without_codegen}" == "${group_dirname}" ]]; then
				found=1
			fi
		done
		if [[ "${found}" -ne "1" && -f "${group_dirname}/types.go" ]] ; then
			echo "need to add ${group_dirname}/ to ${client_gen_file}"
			exit 1
		fi
	fi
done

# import_known_versions checks to be sure we'll get installed
# groups_without_codegen is the list of group we EXPECT to not have the client generated for
# them.  This happens for types that aren't served from the API server
packages_without_install=(
	"k8s.io/kubernetes/pkg/apis/abac"
	"k8s.io/kubernetes/pkg/apis/admission"
)
known_version_files=(
	"pkg/master/import_known_versions.go"
	"pkg/client/clientset_generated/internal_clientset/import_known_versions.go"
)
for expected_install_package in "${expected_install_packages[@]}"; do
	found=0
	for package_without_install in "${packages_without_install[@]}"; do
		if [ "${package_without_install}" == "${expected_install_package}" ]; then
			found=1
		fi
	done
	if [[ "${found}" -eq "1" ]] ; then
		continue
	fi

	for known_version_file in "${known_version_files[@]}"; do
		if ! grep -q "${expected_install_package}/install" ${known_version_files} ; then
			echo "missing ${expected_install_package}/install from ${known_version_files}"
			exit 1
		fi
	done
done

# check all groupversions to make sure they're in the init.sh file.  This isn't perfect, but its slightly
# better than nothing
for external_group_version in "${external_group_versions[@]}"; do
	if ! grep -q "${external_group_version}" "${KUBE_ROOT}/hack/lib/init.sh" ; then
		echo "missing ${external_group_version} from hack/lib/init.sh:/KUBE_AVAILABLE_GROUP_VERSIONS or hack/init.sh:/KUBE_NONSERVER_GROUP_VERSIONS"
		exit 1
	fi
done
