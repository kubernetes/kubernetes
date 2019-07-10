#!/usr/bin/env bash

# Copyright 2017 The Kubernetes Authors.
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

# This script contains the helper functions that each provider hosting
# Kubermark must implement to use test/kubemark/start-kubemark.sh and
# test/kubemark/stop-kubemark.sh scripts.

# This function should authenticate docker to be able to read/write to
# the right container registry (needed for pushing kubemark image).
function authenticate-docker {
	echo "Configuring registry authentication" 1>&2
}

# This function should create kubemark master and write kubeconfig to
# "${RESOURCE_DIRECTORY}/kubeconfig.kubemark".
function create-kubemark-master {
  echo "Creating cluster..."
}

# This function should delete kubemark master.
function delete-kubemark-master {
  echo "Deleting cluster..."
}

# Common colors used throughout the kubemark scripts
if [[ -z "${color_start-}" ]]; then
  declare -r color_start="\033["
  # shellcheck disable=SC2034
  declare -r color_red="${color_start}0;31m"
  # shellcheck disable=SC2034
  declare -r color_yellow="${color_start}0;33m"
  # shellcheck disable=SC2034
  declare -r color_green="${color_start}0;32m"
  # shellcheck disable=SC2034
  declare -r color_blue="${color_start}1;34m"
  # shellcheck disable=SC2034
  declare -r color_cyan="${color_start}1;36m"
  # shellcheck disable=SC2034
  declare -r color_norm="${color_start}0m"
fi
