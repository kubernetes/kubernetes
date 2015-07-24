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

function generate_version() {
	local version=$1
	local TMPFILE="/tmp/conversion_generated.$(date +%s).go"

	echo "Generating for version ${version}"

	sed 's/YEAR/2015/' hooks/boilerplate.go.txt > $TMPFILE
	cat >> $TMPFILE <<EOF
package ${version}

import (
	"reflect"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/conversion"
)

// AUTO-GENERATED FUNCTIONS START HERE
EOF

	GOPATH=$(godep path):$GOPATH go run cmd/genconversion/conversion.go -v ${version} -f - >>  $TMPFILE

	cat >> $TMPFILE <<EOF
// AUTO-GENERATED FUNCTIONS END HERE
EOF

	mv $TMPFILE pkg/api/${version}/conversion_generated.go
}

VERSIONS="v1"
for ver in $VERSIONS; do
  # Ensure that the version being processed is registered by setting
  # KUBE_API_VERSIONS.
  KUBE_API_VERSIONS="${ver}" generate_version "${ver}"
done
