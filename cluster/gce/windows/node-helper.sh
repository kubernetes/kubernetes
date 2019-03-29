#!/usr/bin/env bash

# Copyright 2019 The Kubernetes Authors.
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

# A library of helper functions and constants for Windows nodes.

function get-windows-node-instance-metadata-from-file {
  local metadata=""
  metadata+="cluster-name=${KUBE_TEMP}/cluster-name.txt,"
  metadata+="kube-env=${KUBE_TEMP}/windows-node-kube-env.yaml,"
  metadata+="kubelet-config=${KUBE_TEMP}/windows-node-kubelet-config.yaml,"
  # To get startup script output run "gcloud compute instances
  # get-serial-port-output <instance>" from the location where you're running
  # kube-up.
  metadata+="windows-startup-script-ps1=${KUBE_ROOT}/cluster/gce/windows/configure.ps1,"
  metadata+="common-psm1=${KUBE_ROOT}/cluster/gce/windows/common.psm1,"
  metadata+="k8s-node-setup-psm1=${KUBE_ROOT}/cluster/gce/windows/k8s-node-setup.psm1,"
  metadata+="install-ssh-psm1=${KUBE_ROOT}/cluster/gce/windows/testonly/install-ssh.psm1,"
  metadata+="user-profile-psm1=${KUBE_ROOT}/cluster/gce/windows/testonly/user-profile.psm1,"
  metadata+="${NODE_EXTRA_METADATA}"
  echo "${metadata}"
}

function get-windows-node-instance-metadata {
  local metadata=""
  metadata+="k8s-version=${KUBE_VERSION:-v1.13.2},"
  # Prevent the GCE Windows agent from managing IP addresses, since kube-proxy
  # and these cluster setup scripts should take care of everything. See
  # https://github.com/kubernetes/kubernetes/issues/75561.
  metadata+="disable-address-manager=true,"
  metadata+="serial-port-enable=1,"
  # This enables logging the serial port output.
  # https://cloud.google.com/compute/docs/instances/viewing-serial-port-output
  metadata+="serial-port-logging-enable=true,"
  metadata+="win-version=${WINDOWS_NODE_OS_DISTRIBUTION}"
  echo "${metadata}"
}

# $1: template name (required).
# $2: scopes flag.
function create-windows-node-instance-template {
  local template_name="$1"
  local scopes_flag="$2"
  create-node-template "${template_name}" "${scopes_flag}" "$(get-windows-node-instance-metadata-from-file)" "$(get-windows-node-instance-metadata)" "windows"
}
