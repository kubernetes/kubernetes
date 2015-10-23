#!/bin/bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
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

# A single sciprt that runs a predefined set of update-* scripts, as they often go together.
set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/cluster/kube-env.sh"

SILENT=true

while getopts ":v" opt; do
	case $opt in
		v)
			SILENT=false
			;;
		\?)
			echo "Invalid flag: -$OPTARG" >&2
			exit 1
			;;
	esac
done

if $SILENT ; then
	echo "Running in the silent mode, run with -v if you want to see script logs."
fi

BASH_TARGETS="codecgen
	generated-conversions
	generated-deep-copies 
	generated-docs 
	generated-swagger-docs 
	swagger-spec
	api-reference-docs"


for t in $BASH_TARGETS
do
	echo -e "Updating $t"
	if $SILENT ; then
		bash "$KUBE_ROOT/hack/update-$t.sh" 1> /dev/null || echo -e "${color_red}FAILED${color_norm}" 
	else
		bash "$KUBE_ROOT/hack/update-$t.sh"
	fi
done
