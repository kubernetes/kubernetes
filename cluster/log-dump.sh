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

# Call this to dump all master and node logs into the folder specified in $1
# (defaults to _artifacts). Only works if the provider supports SSH.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
: ${KUBE_CONFIG_FILE:="config-test.sh"}

source "${KUBE_ROOT}/cluster/kube-util.sh"
detect-project &> /dev/null

readonly report_dir="${1:-_artifacts}"
echo "Dumping master and node logs to ${report_dir}"

# Copy all files /var/log/{$3}.log on node $1 into local dir $2.
# $3 should be a space-separated string of files.
# This function shouldn't ever trigger errexit, but doesn't block stderr.
function copy-logs-from-node() {
    local -r node="${1}"
    local -r dir="${2}"
    local files=(${3})
    # Append ".log"
    files=("${files[@]/%/.log}")
    # Prepend "/var/log/"
    files=("${files[@]/#/\/var\/log\/}")
    # Replace spaces with commas, surround with braces
    local -r scp_files="{$(echo ${files[*]} | tr ' ' ',')}"

    if [[ "${KUBERNETES_PROVIDER}" == "aws" ]]; then
        local ip=$(get_ssh_hostname "${node}")
        scp -oLogLevel=quiet -oConnectTimeout=30 -oStrictHostKeyChecking=no -i "${AWS_SSH_KEY}" "${SSH_USER}@${ip}:${scp_files}" "${dir}" > /dev/null || true
    else
        gcloud compute copy-files --project "${PROJECT}" --zone "${ZONE}" "${node}:${scp_files}" "${dir}" > /dev/null || true
    fi
}

# Save logs for node $1 into directory $2. Pass in any non-common files in $3.
# $3 should be a space-separated list of files.
# This function shouldn't ever trigger errexit
function save-logs() {
    local -r node_name="${1}"
    local -r dir="${2}"
    local files="${3} ${common_logfiles}"
    if [[ "${KUBERNETES_PROVIDER}" == "gce" ]]; then
        files="${files} ${gce_logfiles}"
    fi
    if [[ "${KUBERNETES_PROVIDER}" == "aws" ]]; then
        files="${files} ${aws_logfiles}"
    fi
    if ssh-to-node "${node_name}" "sudo systemctl status kubelet.service" &> /dev/null; then
        ssh-to-node "${node_name}" "sudo journalctl --output=cat -u kubelet.service" > "${dir}/kubelet.log" || true
        ssh-to-node "${node_name}" "sudo journalctl --output=cat -u docker.service" > "${dir}/docker.log" || true
    else
        files="${files} ${initd_logfiles} ${supervisord_logfiles}"
    fi
    copy-logs-from-node "${node_name}" "${dir}" "${files}"
}

readonly master_ssh_supported_providers="gce aws kubemark"
readonly node_ssh_supported_providers="gce gke aws"

readonly master_logfiles="kube-apiserver kube-scheduler kube-controller-manager etcd glbc cluster-autoscaler"
readonly node_logfiles="kube-proxy"
readonly aws_logfiles="cloud-init-output"
readonly gce_logfiles="startupscript"
readonly common_logfiles="kern"
readonly initd_logfiles="docker"
readonly supervisord_logfiles="kubelet supervisor/supervisord supervisor/kubelet-stdout supervisor/kubelet-stderr supervisor/docker-stdout supervisor/docker-stderr"

# Limit the number of concurrent node connections so that we don't run out of
# file descriptors for large clusters.
readonly max_scp_processes=25

if [[ ! "${master_ssh_supported_providers}" =~ "${KUBERNETES_PROVIDER}" ]]; then
    echo "Master SSH not supported for ${KUBERNETES_PROVIDER}"
elif ! (detect-master &> /dev/null); then
    echo "Master not detected. Is the cluster up?"
else
    readonly master_dir="${report_dir}/${MASTER_NAME}"
    mkdir -p "${master_dir}"
    save-logs "${MASTER_NAME}" "${master_dir}" "${master_logfiles}"
fi

detect-node-names &> /dev/null
if [[ ! "${node_ssh_supported_providers}"  =~ "${KUBERNETES_PROVIDER}" ]]; then
    echo "Node SSH not supported for ${KUBERNETES_PROVIDER}"
elif [[ "${#NODE_NAMES[@]}" -eq 0 ]]; then
    echo "Nodes not detected. Is the cluster up?"
else
    proc=${max_scp_processes}
    for node_name in "${NODE_NAMES[@]}"; do
        node_dir="${report_dir}/${node_name}"
        mkdir -p "${node_dir}"
        # Save logs in the background. This speeds up things when there are
        # many nodes.
        save-logs "${node_name}" "${node_dir}" "${node_logfiles}" &
        # We don't want to run more than ${max_scp_processes} at a time, so
        # wait once we hit that many nodes. This isn't ideal, since one might
        # take much longer than the others, but it should help.
        proc=$((proc - 1))
        if [[ proc -eq 0 ]]; then
            proc=${max_scp_processes}
            wait
        fi
    done
    # Wait for any remaining processes.
    if [[ proc -gt 0 && proc -lt ${max_scp_processes} ]]; then
        wait
    fi
fi
