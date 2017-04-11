#!/bin/bash

# Copyright 2015 The Kubernetes Authors.
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

ACTION=${1}
SERVICE=${2}

if [[ -z "${ACTION}" || -z "${SERVICE}" ]]; then
  echo "Syntax: ${0} <action> <service>"
  exit 1
fi


function reload_state() {
  systemctl daemon-reload
}

function start_service() {
  systemctl start ${SERVICE}
}

function stop_service() {
  systemctl stop ${SERVICE}
}

function enable_service() {
  systemctl enable ${SERVICE}
}

function disable_service() {
  systemctl disable ${SERVICE}
}

function restart_service() {
  systemctl restart ${SERVICE}
}

if [[ "${ACTION}" == "up" ]]; then
  reload_state
  enable_service
  start_service
elif [[ "${ACTION}" == "bounce" ]]; then
  reload_state
  enable_service
  restart_service
elif [[ "${ACTION}" == "down" ]]; then
  reload_state
  disable_service
  stop_service
elif [[ "${ACTION}" == "enable" ]]; then
  reload_state
  enable_service
else
  echo "Unknown action: ${ACTION}"
  exit 1
fi
