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

# A library of helper functions and constant for GCI distro

# Creates the GCI specific metadata files if they do not exit.
# Assumed var
#   KUBE_TEMP
function ensure-gci-metadata-files {
  if [[ ! -f "${KUBE_TEMP}/gci-update.txt" ]]; then
    cat >"${KUBE_TEMP}/gci-update.txt" << EOF
update_disabled
EOF
  fi
  if [[ ! -f "${KUBE_TEMP}/gci-ensure-gke-docker.txt" ]]; then
    cat >"${KUBE_TEMP}/gci-ensure-gke-docker.txt" << EOF
true
EOF
  fi
  if [[ ! -f "${KUBE_TEMP}/gci-docker-version.txt" ]]; then
    cat >"${KUBE_TEMP}/gci-docker-version.txt" << EOF
${GCI_DOCKER_VERSION:-}
EOF
  fi
}
