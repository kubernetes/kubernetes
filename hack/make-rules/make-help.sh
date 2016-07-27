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

set -o errexit
set -o nounset
set -o pipefail

red='\E[1;31m'
reset='\E[0m'

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
ALL_TARGETS=$(make -C "${KUBE_ROOT}" PRINT_HELP=y -rpn | sed -n -e '/^$/ { n ; /^[^ .#][^ ]*:/ { s/:.*$// ; p ; } ; }' | sort)
CMD_TARGETS=$(ls -l "${KUBE_ROOT}/cmd" |awk '/^d/ {print $NF}')
PLUGIN_CMD_TARGETS=$(ls -l "${KUBE_ROOT}/plugin/cmd" |awk '/^d/ {print $NF}')
FED_CMD_TARGETS=$(ls -l "${KUBE_ROOT}/federation/cmd" |awk '/^d/ {print $NF}')
CMD_FLAG=false
PLUGIN_CMD_FLAG=false
FED_CMD_FLAG=false

echo "--------------------------------------------------------------------------------"
for tar in $ALL_TARGETS; do
	for cmdtar in $CMD_TARGETS; do
		if [ $tar = $cmdtar ]; then
			if [ $CMD_FLAG = true ]; then
				continue 2;
			fi

			echo -e "${red}${CMD_TARGETS}${reset}"
			make -C "${KUBE_ROOT}" $tar PRINT_HELP=y
			echo "---------------------------------------------------------------------------------"

			CMD_FLAG=true
			continue 2
		fi
	done

	for plugincmdtar in $PLUGIN_CMD_TARGETS; do
		if [ $tar = $plugincmdtar ]; then
			if [ $PLUGIN_CMD_FLAG = true ]; then
				continue 2;
			fi

			echo -e "${red}${PLUGIN_CMD_TARGETS}${reset}"
			make -C "${KUBE_ROOT}" $tar PRINT_HELP=y
			echo "---------------------------------------------------------------------------------"

			PLUGIN_CMD_FLAG=true
			continue 2
		fi
	done

	for fedcmdtar in $FED_CMD_TARGETS; do
		if [ $tar = $fedcmdtar ]; then
			if [ $FED_CMD_FLAG = true ]; then
				continue 2;
			fi

			echo -e "${red}${FED_CMD_TARGETS}${reset}"
			make -C "${KUBE_ROOT}" $tar PRINT_HELP=y
			echo "---------------------------------------------------------------------------------"

			FED_CMD_FLAG=true
			continue 2
		fi
	done

	echo -e "${red}${tar}${reset}"
	make -C "${KUBE_ROOT}" $tar PRINT_HELP=y
	echo "---------------------------------------------------------------------------------"
done
