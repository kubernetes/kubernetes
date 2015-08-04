#!/bin/bash

# Copyright 2015 The Kubernetes Authors All rights reserved.
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

function result_file_name() {
	local version=$1
	echo "pkg/${version}/deep_copy_generated.go"
}

function generate_version() {
	local version=$1
	local genunit=$2
	local TMPFILE="/tmp/deep_copy_generated.$(date +%s).go"

	echo "Generating for version ${version}"

	pkgname=${version##*/}
	if [[ -z $pkgname ]]; then
		pkgname=${version%/*}
	fi

	sed 's/YEAR/2015/' hooks/boilerplate.go.txt > "$TMPFILE"
	cat >> "$TMPFILE" <<EOF
package $pkgname

// AUTO-GENERATED FUNCTIONS START HERE
EOF

	env GOPATH=$(godep path):$GOPATH go run cmd/gendeepcopy/deep_copy.go -v "${version}" -u "${genunit}" -f - -o "${version}=" >>  $TMPFILE

	cat >> "$TMPFILE" <<EOF
// AUTO-GENERATED FUNCTIONS END HERE
EOF

	env GOPATH=$(godep path):$GOPATH goimports -w "$TMPFILE"
	mv $TMPFILE `result_file_name ${version}`
}

function generate_deep_copies() {
	local version_genunits=(
		"api/" "github.com/GoogleCloudPlatform/kubernetes"
		"api/v1" "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1"
		"expapi/" "github.com/GoogleCloudPlatform/kubernetes/pkg/expapi"
		"expapi/v1" "github.com/GoogleCloudPlatform/kubernetes/pkg/expapi/v1"
	)
	# To avoid compile errors, remove the currently existing files.
	for ((i=0; i < ${#version_genunits[@]}; i+=2)); do
		rm -f `result_file_name ${version_genunits[i]}`
	done
	for ((i=0; i < ${#version_genunits[@]}; i+=2)); do
		# Ensure that the version being processed is registered by setting
		# KUBE_API_VERSIONS.
		apiVersions="${version_genunits[i]##*/}"
		KUBE_API_VERSIONS="${apiVersions}" generate_version "${version_genunits[i]}" "${version_genunits[i+1]}"
	done
}

generate_deep_copies
