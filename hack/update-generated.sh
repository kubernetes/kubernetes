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

# Script to fetch latest swagger spec.
# Puts the updated spec at swagger-spec/

set -o errexit
set -o nounset
set -o pipefail

function generate_version() {
local version=$1

echo "Generating for version ${version}"

cat > /tmp/conversion_generated.go <<EOF
/*
Copyright 2015 The Kubernetes Authors All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package ${version}

import (
	"reflect"

	newer "github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/conversion"
)

// AUTO-GENERATED FUNCTIONS START HERE
EOF


go run cmd/kube-conversion/conversion.go -v ${version} -f - -n /dev/null >>  /tmp/conversion_generated.go

cat >> /tmp/conversion_generated.go <<EOF
// AUTO-GENERATED FUNCTIONS END HERE

func init() {
	err := newer.Scheme.AddGeneratedConversionFuncs(
EOF

go run cmd/kube-conversion/conversion.go -v ${version} -f /dev/null -n - >>  /tmp/conversion_generated.go

cat >> /tmp/conversion_generated.go <<EOF
	)
	if err != nil {
		// If one of the conversion functions is malformed, detect it immediately.
		panic(err)
	}
}
EOF

mv /tmp/conversion_generated.go pkg/api/${version}/conversion_generated.go

}

generate_version v1
generate_version v1beta3
