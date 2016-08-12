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

kube::golang::verify_go_version

cd "${KUBE_ROOT}"

array_contains () {
    local seeking=$1; shift # shift will iterate through the array
    local in=1 # in holds the exit status for the function
    for element; do
        if [[ "$element" == "$seeking" ]]; then
            in=0 # set in to 0 since we found it
            break
        fi
    done
    return $in
}

export IFS=$'\n'
all_packages=(
	$(go list -e ./... | egrep -v "/(third_party|vendor|staging|generated|clientset_generated)" | sed 's/k8s.io\/kubernetes\///g')
)
linted_file="${KUBE_ROOT}/hack/.linted_packages"
linted_packages=(
	$(cat $linted_file)
)
unset IFS
linted=()
errors=()
for p in "${all_packages[@]}"; do
	# Run golint on package/*.go file explicitly to validate all go files
	# and not just the ones for the current platform.
	failedLint=$(golint "$p"/*.go)
	if [ "$failedLint" ]; then
		if array_contains "$p" "${linted_packages[@]}"; then
			errors+=( "$failedLint" )
		fi
	else
		array_contains "$p" "${linted_packages[@]}" || linted+=( "$p" )
	fi
done

# Check to be sure all the packages that should pass lint are.
if [ ${#errors[@]} -eq 0 ]; then
	echo 'Congratulations!  All Go source files have been linted.'
else
	{
		echo "Errors from golint:"
		for err in "${errors[@]}"; do
			echo "$err"
		done
		echo
		echo 'Please fix the above errors. You can test via "golint" and commit the result.'
		echo
	} >&2
	false
fi

# check to make sure all packages that pass lint are in the linted file.
if [ ${#linted[@]} -eq 0 ]; then
	echo 'Success! All packages that should pass lint are listed in the linted file.'
else
	{
		echo "The following packages passed golint but are not listed in $linted_file:"
		for p in "${linted[@]}"; do
			echo "echo $p >> hack/.linted_packages"
		done
		echo
		echo 'Please add the following packages to the linted file. You can test via this script and commit the result.'
		echo
	} >&2
	false
fi
