#!/bin/bash

# Copyright 2016 The Kubernetes Authors All rights reserved.
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

# Call this to dump all master and node logs into the folder specified in $1
# (defaults to _artifacts). Only works if the provider supports SSH.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
: ${KUBE_CONFIG_FILE:="config-test.sh"}

source "${KUBE_ROOT}/cluster/kube-env.sh"
source "${KUBE_ROOT}/cluster/kube-util.sh"

readonly report_dir="${1:-_artifacts}"
echo "Dumping master and node logs to ${report_dir}"

# Saves a single output of running a given command ($2) on a given node ($1)
# into a given local file ($3). Does not fail if the ssh command fails for any
# reason, just prints an error to stderr.
function save-log() {
    local -r node_name="${1}"
    local -r cmd="${2}"
    local -r output_file="${3}"
    if ! ssh-to-node "${node_name}" "${cmd}" > "${output_file}"; then
        echo "${cmd} failed for ${node_name}" >&2
    fi
}

# Saves logs common to master and nodes. The node name is in $1 and the
# directory/name prefix is in $2. Assumes KUBERNETES_PROVIDER is set.
function save-common-logs() {
    local -r node_name="${1}"
    local -r prefix="${2}"
    echo "Dumping logs for ${node_name}"
    save-log "${node_name}" "cat /var/log/kern.log" "${prefix}-kern.log"
    save-log "${node_name}" "cat /var/log/docker.log" "${prefix}-docker.log"
    if [[ "${KUBERNETES_PROVIDER}" == "gce" ]]; then
        save-log "${node_name}" "cat /var/log/startupscript.log" "${prefix}-startupscript.log"
    fi
    if ssh-to-node "${node_name}" "sudo systemctl status kubelet.service" &> /dev/null; then
        save-log "${node_name}" "sudo journalctl --output=cat -u kubelet.service" "${prefix}-kubelet.log"
    else
        save-log "${node_name}" "cat /var/log/kubelet.log" "${prefix}-kubelet.log"
        save-log "${node_name}" "cat /var/log/supervisor/supervisord.log" "${prefix}-supervisord.log"
        save-log "${node_name}" "cat /var/log/supervisor/kubelet-stdout.log" "${prefix}-supervisord-kubelet-stdout.log"
        save-log "${node_name}" "cat /var/log/supervisor/kubelet-stderr.log" "${prefix}-supervisord-kubelet-stderr.log"
    fi
}

readonly master_ssh_supported_providers="gce aws kubemark"
readonly node_ssh_supported_providers="gce gke aws"

if [[ ! "${master_ssh_supported_providers}" =~ "${KUBERNETES_PROVIDER}" ]]; then
    echo "Master SSH not supported for ${KUBERNETES_PROVIDER}"
elif ! $(detect-master &> /dev/null); then
    echo "Master not detected. Is the cluster up?"
else
    echo "Master Name: ${MASTER_NAME}"
    readonly master_prefix="${report_dir}/${MASTER_NAME}"
    save-log "${MASTER_NAME}" "cat /var/log/kube-apiserver.log" "${master_prefix}-kube-apiserver.log"
    save-log "${MASTER_NAME}" "cat /var/log/kube-scheduler.log" "${master_prefix}-kube-scheduler.log"
    save-log "${MASTER_NAME}" "cat /var/log/kube-controller-manager.log" "${master_prefix}-kube-controller-manager.log"
    save-log "${MASTER_NAME}" "cat /var/log/etcd.log" "${master_prefix}-kube-etcd.log"
    save-common-logs "${MASTER_NAME}" "${master_prefix}"
fi

detect-node-names &> /dev/null
if [[ ! "${node_ssh_supported_providers}"  =~ "${KUBERNETES_PROVIDER}" ]]; then
    echo "Node SSH not supported for ${KUBERNETES_PROVIDER}"
elif [[ "${#NODE_NAMES[@]}" -eq 0 ]]; then
    echo "Nodes not detected. Is the cluster up?"
else
    echo "Node Names: ${NODE_NAMES[*]}"
    for node_name in "${NODE_NAMES[@]}"; do
        node_prefix="${report_dir}/${node_name}"
        save-log "${node_name}" "cat /var/log/kube-proxy.log" "${node_prefix}-kube-proxy.log"
        save-common-logs "${node_name}" "${node_prefix}"
    done
fi
