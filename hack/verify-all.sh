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

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/cluster/kube-env.sh"

SILENT=true

function is-excluded {
	for e in $EXCLUDE; do
		if [[ $1 -ef ${BASH_SOURCE} ]]; then
			return
		fi
		if [[ $1 -ef "$KUBE_ROOT/hack/$e" ]]; then
			return
		fi
	done
	return 1
}

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

EXCLUDE="verify-godeps.sh"

for t in `ls $KUBE_ROOT/hack/verify-*.sh`
do
	if is-excluded $t ; then
		echo "Skipping $t"
		continue
	fi
	if $SILENT ; then
		echo -e "Verifying $t"
		bash "$t" &> /dev/null && echo -e "${color_green}SUCCESS${color_norm}" || echo -e "${color_red}FAILED${color_norm}" 
	else
		bash "$t" || true
	fi
done

for t in `ls $KUBE_ROOT/hack/verify-*.py`
do
	if is-excluded $t ; then
		echo "Skipping $t"
		continue
	fi
	if $SILENT ; then
		echo -e "Verifying $t"
		python "$t" &> /dev/null && echo -e "${color_green}SUCCESS${color_norm}" || echo -e "${color_red}FAILED${color_norm}" 
	else 
		python "$t" || true
	fi
done