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

set -o errexit
set -o nounset
set -o pipefail

red=$(tput setaf 1)
reset=$(tput sgr0)
readonly red reset

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/../..
ALL_TARGETS=$(make -C "${KUBE_ROOT}" PRINT_HELP=y -rpn | sed -n -e '/^$/ { n ; /^[^ .#][^ ]*:/ { s/:.*$// ; p ; } ; }' | sort)
CMD_TARGETS=$(cd "${KUBE_ROOT}/cmd"; find . -mindepth 1 -maxdepth 1 -type d | cut -c 3-)
CMD_FLAG=false

echo "--------------------------------------------------------------------------------"
for tar in ${ALL_TARGETS}; do
	for cmdtar in ${CMD_TARGETS}; do
		if [ "${tar}" = "${cmdtar}" ]; then
			if [ ${CMD_FLAG} = true ]; then
				continue 2;
			fi

			echo -e "${red}${CMD_TARGETS}${reset}"
			make -C "${KUBE_ROOT}" "${tar}" PRINT_HELP=y
			echo "---------------------------------------------------------------------------------"

			CMD_FLAG=true
			continue 2
		fi
	done

	echo -e "${red}${tar}${reset}"
	make -C "${KUBE_ROOT}" "${tar}" PRINT_HELP=y
	echo "---------------------------------------------------------------------------------"
done
