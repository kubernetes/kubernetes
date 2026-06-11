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

# Call this to dump all master and node logs into the folder specified in $1
# (defaults to _artifacts). Only works if the provider supports SSH.

set -o errexit
set -o nounset
set -o pipefail

readonly local_report_dir="${1:-_artifacts}"
report_dir=""
readonly gcs_artifacts_dir="${2:-}"
readonly logexporter_namespace="${3:-logexporter}"

# In order to more trivially extend log-dump for custom deployments,
# check for a function named log_dump_custom_get_instances. If it's
# defined, we assume the function can me called with one argument, the
# role, which is either "master" or "node".
echo 'Checking for custom logdump instances, if any'
if [[ $(type -t log_dump_custom_get_instances) == "function" ]]; then
  readonly use_custom_instance_list=yes
else
  readonly use_custom_instance_list=
fi

readonly master_ssh_supported_providers="gce aws"
readonly gcloud_supported_providers="gce gke"

readonly master_logfiles="kube-apiserver.log kube-apiserver-audit.log kube-scheduler.log cloud-controller-manager.log kube-controller-manager.log etcd.log etcd-events.log glbc.log cluster-autoscaler.log kube-addon-manager.log konnectivity-server.log fluentd.log kubelet.cov"
readonly node_logfiles="kube-proxy.log containers/konnectivity-agent-*.log fluentd.log node-problem-detector.log kubelet.cov"
readonly node_systemd_services="node-problem-detector"
readonly hollow_node_logfiles="kubelet-hollow-node-*.log kubeproxy-hollow-node-*.log npd-hollow-node-*.log"
readonly aws_logfiles="cloud-init-output.log"
readonly gce_logfiles="startupscript.log"
readonly kern_logfile="kern.log"
readonly initd_logfiles="docker/log"
readonly supervisord_logfiles="kubelet.log supervisor/supervisord.log supervisor/kubelet-stdout.log supervisor/kubelet-stderr.log supervisor/docker-stdout.log supervisor/docker-stderr.log"
readonly systemd_services="kubelet kubelet-monitor kube-container-runtime-monitor ${LOG_DUMP_SYSTEMD_SERVICES:-containerd}"
readonly extra_log_files="${LOG_DUMP_EXTRA_FILES:-}"
readonly extra_systemd_services="${LOG_DUMP_SAVE_SERVICES:-}"
readonly dump_systemd_journal="${LOG_DUMP_SYSTEMD_JOURNAL:-false}"

# Root directory for Kubernetes files on Windows nodes.
WINDOWS_K8S_DIR="C:\\etc\\kubernetes"
# Directory where Kubernetes log files will be stored on Windows nodes.
export WINDOWS_LOGS_DIR="${WINDOWS_K8S_DIR}\\logs"
# Log files found in WINDOWS_LOGS_DIR on Windows nodes:
readonly windows_node_logfiles="kubelet.log kube-proxy.log docker.log docker_images.log csi-proxy.log"
# Log files found in other directories on Windows nodes:
readonly windows_node_otherfiles="C:\\Windows\\MEMORY.dmp"

# Limit the number of concurrent node connections so that we don't run out of
# file descriptors for large clusters.
readonly max_dump_processes=25

# Indicator variable whether we experienced a significant failure during
# logexporter creation or execution.
logexporter_failed=0

# Percentage of nodes that must be logexported successfully (otherwise the
# process will exit with a non-zero exit code).
readonly log_dump_expected_success_percentage="${LOG_DUMP_EXPECTED_SUCCESS_PERCENTAGE:-0}"

# Use the gcloud defaults to find the project.  If it is already set in the
# environment then go with that.
#
# Vars set:
#   PROJECT
#   NETWORK_PROJECT
#   PROJECT_REPORTED
function detect-project() {
  if [[ -z "${PROJECT-}" ]]; then
    PROJECT=$(gcloud config list project --format 'value(core.project)')
  fi

  NETWORK_PROJECT=${NETWORK_PROJECT:-${PROJECT}}

  if [[ -z "${PROJECT-}" ]]; then
    echo "Could not detect Google Cloud Platform project.  Set the default project using " >&2
    echo "'gcloud config set project <PROJECT>'" >&2
    exit 1
  fi
  if [[ -z "${PROJECT_REPORTED-}" ]]; then
    echo "Project: ${PROJECT}" >&2
    echo "Network Project: ${NETWORK_PROJECT}" >&2
    echo "Zone: ${ZONE}" >&2
    PROJECT_REPORTED=true
  fi
}

# Detect Linux and Windows nodes in the cluster.
#
# If a custom get-instances function has been set, this function will use it
# to set the NODE_NAMES array.
#
# Otherwise this function will attempt to detect the nodes based on the GCP
# instance group information. If Windows nodes are present they will be detected
# separately. The following arrays will be set:
#   NODE_NAMES
#   INSTANCE_GROUPS
#   WINDOWS_NODE_NAMES
#   WINDOWS_INSTANCE_GROUPS
function detect-node-names() {
  NODE_NAMES=()
  INSTANCE_GROUPS=()
  WINDOWS_INSTANCE_GROUPS=()
  WINDOWS_NODE_NAMES=()

  if [[ -n "${use_custom_instance_list}" ]]; then
    echo 'Detecting node names using log_dump_custom_get_instances() function'
    while IFS='' read -r line; do NODE_NAMES+=("$line"); done < <(log_dump_custom_get_instances node)
    echo "NODE_NAMES=${NODE_NAMES[*]:-}" >&2
    return
  fi

  if ! [[ "${gcloud_supported_providers}" =~ ${KUBERNETES_PROVIDER} ]]; then
    echo "gcloud not supported for ${KUBERNETES_PROVIDER}, can't detect node names"
    return
  fi

  # These prefixes must not be prefixes of each other, so that they can be used to
  # detect mutually exclusive sets of nodes.
  local -r NODE_INSTANCE_PREFIX=${NODE_INSTANCE_PREFIX:-"${INSTANCE_PREFIX}-minion"}
  local -r WINDOWS_NODE_INSTANCE_PREFIX=${WINDOWS_NODE_INSTANCE_PREFIX:-"${INSTANCE_PREFIX}-windows-node"}
  detect-project
  echo 'Detecting nodes in the cluster'
  INSTANCE_GROUPS+=($(gcloud compute instance-groups managed list \
    --project "${PROJECT}" \
    --filter "name ~ '${NODE_INSTANCE_PREFIX}-.+' AND zone:(${ZONE})" \
    --format='value(name)' || true))
  WINDOWS_INSTANCE_GROUPS+=($(gcloud compute instance-groups managed list \
    --project "${PROJECT}" \
    --filter "name ~ '${WINDOWS_NODE_INSTANCE_PREFIX}-.+' AND zone:(${ZONE})" \
    --format='value(name)' || true))

  if [[ -n "${INSTANCE_GROUPS[@]:-}" ]]; then
    for group in "${INSTANCE_GROUPS[@]}"; do
      NODE_NAMES+=($(gcloud compute instance-groups managed list-instances \
        "${group}" --zone "${ZONE}" --project "${PROJECT}" \
        --format='value(name)'))
    done
  fi
  # Add heapster node name to the list too (if it exists).
  if [[ -n "${HEAPSTER_MACHINE_TYPE:-}" ]]; then
    NODE_NAMES+=("${NODE_INSTANCE_PREFIX}-heapster")
  fi
  if [[ -n "${WINDOWS_INSTANCE_GROUPS[@]:-}" ]]; then
    for group in "${WINDOWS_INSTANCE_GROUPS[@]}"; do
      WINDOWS_NODE_NAMES+=($(gcloud compute instance-groups managed \
        list-instances "${group}" --zone "${ZONE}" --project "${PROJECT}" \
        --format='value(name)'))
    done
  fi

  echo "INSTANCE_GROUPS=${INSTANCE_GROUPS[*]:-}" >&2
  echo "NODE_NAMES=${NODE_NAMES[*]:-}" >&2
  echo "WINDOWS_INSTANCE_GROUPS=${WINDOWS_INSTANCE_GROUPS[*]:-}" >&2
  echo "WINDOWS_NODE_NAMES=${WINDOWS_NODE_NAMES[*]:-}" >&2
}

# Detect the IP for the master
#
# Assumed vars:
#   MASTER_NAME
#   ZONE
#   REGION
# Vars set:
#   KUBE_MASTER
#   KUBE_MASTER_IP
function detect-master() {
  detect-project
  KUBE_MASTER=${MASTER_NAME}
  echo "Trying to find master named '${MASTER_NAME}'" >&2
  if [[ -z "${KUBE_MASTER_IP-}" ]]; then
    local master_address_name="${MASTER_NAME}-ip"
    echo "Looking for address '${master_address_name}'" >&2
    if ! KUBE_MASTER_IP=$(gcloud compute addresses describe "${master_address_name}" \
      --project "${PROJECT}" --region "${REGION}" -q --format='value(address)') || \
      [[ -z "${KUBE_MASTER_IP-}" ]]; then
      echo "Could not detect Kubernetes master node.  Make sure you've launched a cluster with 'kube-up.sh'" >&2
      exit 1
    fi
  fi
  if [[ -z "${KUBE_MASTER_INTERNAL_IP-}" ]] && [[ ${GCE_PRIVATE_CLUSTER:-} == "true" ]]; then
      local master_address_name="${MASTER_NAME}-internal-ip"
      echo "Looking for address '${master_address_name}'" >&2
      if ! KUBE_MASTER_INTERNAL_IP=$(gcloud compute addresses describe "${master_address_name}" \
        --project "${PROJECT}" --region "${REGION}" -q --format='value(address)') || \
        [[ -z "${KUBE_MASTER_INTERNAL_IP-}" ]]; then
        echo "Could not detect Kubernetes master node.  Make sure you've launched a cluster with 'kube-up.sh'" >&2
        exit 1
      fi
  fi
  echo "Using master: $KUBE_MASTER (external IP: $KUBE_MASTER_IP; internal IP: ${KUBE_MASTER_INTERNAL_IP:-(not set)})" >&2
}

# SSH to a node by name ($1) and run a command ($2).
function setup() {
  KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/../..
  if [[ -z "${use_custom_instance_list}" ]]; then
    echo "Using gce provider, skipping check for LOG_DUMP_SSH_KEY and LOG_DUMP_SSH_USER"
    echo 'Sourcing kube-util.sh'
    source "${KUBE_ROOT}/cluster/kube-util.sh"
    ZONE="${KUBE_GCE_ZONE:-us-central1-b}"
    REGION="${ZONE%-*}"
    INSTANCE_PREFIX="${KUBE_GCE_INSTANCE_PREFIX:-kubernetes}"
    CLUSTER_NAME="${CLUSTER_NAME:-${INSTANCE_PREFIX}}"
    MASTER_NAME="${INSTANCE_PREFIX}-master"
    GCE_PRIVATE_CLUSTER="${KUBE_GCE_PRIVATE_CLUSTER:-false}"
    echo 'Detecting project'
    detect-project 2>&1
  elif [[ "${KUBERNETES_PROVIDER}" == "gke" ]]; then
    NUM_NODES=${NUM_NODES:-3}
    echo "Using 'use_custom_instance_list' with gke, skipping check for LOG_DUMP_SSH_KEY and LOG_DUMP_SSH_USER"
  elif [[ -z "${LOG_DUMP_SSH_KEY:-}" ]]; then
    echo 'LOG_DUMP_SSH_KEY not set, but required when using log_dump_custom_get_instances'
    exit 1
  elif [[ -z "${LOG_DUMP_SSH_USER:-}" ]]; then
    echo 'LOG_DUMP_SSH_USER not set, but required when using log_dump_custom_get_instances'
    exit 1
  fi
}

function log-dump-ssh() {
  local host="$1"
  local cmd="$2"

  if [[ "${gcloud_supported_providers}" =~ ${KUBERNETES_PROVIDER} ]]; then
    for (( i=0; i<5; i++)); do
      if gcloud compute ssh --ssh-flag="-o LogLevel=quiet" --ssh-flag="-o ConnectTimeout=30" --project "${PROJECT}" --zone="${ZONE}" "${host}" --command "echo test > /dev/null"; then
        break
      fi
      sleep 5
    done
    # Then actually try the command.
    gcloud compute ssh --ssh-flag="-o LogLevel=quiet" --ssh-flag="-o ConnectTimeout=30" --project "${PROJECT}" --zone="${ZONE}" "${host}" --command "${cmd}"
    return
  fi

  ssh -oLogLevel=quiet -oConnectTimeout=30 -oStrictHostKeyChecking=no -i "${LOG_DUMP_SSH_KEY}" "${LOG_DUMP_SSH_USER}@${host}" "${cmd}"
}

# Copy all files /var/log/{$3}.log on node $1 into local dir $2.
# $3 should be a string array of file names.
# This function shouldn't ever trigger errexit, but doesn't block stderr.
function copy-logs-from-node() {
    local -r node="${1}"
    local -r dir="${2}"
    shift
    shift
    local files=("$@")
    # Append "*"
    # The * at the end is needed to also copy rotated logs (which happens
    # in large clusters and long runs).
    files=( "${files[@]/%/*}" )
    # Prepend "/var/log/"
    files=( "${files[@]/#/\/var\/log\/}" )
    # Comma delimit (even the singleton, or scp does the wrong thing), surround by braces.
    local -r scp_files="{$(printf "%s," "${files[@]}")}"

    if [[ "${gcloud_supported_providers}" =~ ${KUBERNETES_PROVIDER} ]]; then
      # get-serial-port-output lets you ask for ports 1-4, but currently (11/21/2016) only port 1 contains useful information
      gcloud compute instances get-serial-port-output --project "${PROJECT}" --zone "${ZONE}" --port 1 "${node}" > "${dir}/serial-1.log" || true
      # FIXME(dims): bug in gcloud prevents multiple source files specified using curly braces, so we just loop through for now
      for single_file in "${files[@]}"; do
        # gcloud scp doesn't work very well when trying to fetch constantly changing files such as logs, as it blocks forever sometimes.
        # We set ConnectTimeout to 5s to avoid blocking for (default tested on 2023-11-17) 2m.
        gcloud compute ssh --project "${PROJECT}" --zone "${ZONE}" "${node}" --command "tar -zcvf - ${single_file}" -- -o ConnectTimeout=5 | tar -zxf - --strip-components=2 -C "${dir}" || true
      done
    elif  [[ "${KUBERNETES_PROVIDER}" == "aws" ]]; then
      local ip
      ip=$(get_ssh_hostname "${node}")
      scp -oLogLevel=quiet -oConnectTimeout=30 -oStrictHostKeyChecking=no -i "${AWS_SSH_KEY}" "${SSH_USER}@${ip}:${scp_files}" "${dir}" > /dev/null || true
    elif  [[ -n "${use_custom_instance_list}" ]]; then
      scp -oLogLevel=quiet -oConnectTimeout=30 -oStrictHostKeyChecking=no -i "${LOG_DUMP_SSH_KEY}" "${LOG_DUMP_SSH_USER}@${node}:${scp_files}" "${dir}" > /dev/null || true
    else
      echo "Unknown cloud-provider '${KUBERNETES_PROVIDER}' and use_custom_instance_list is unset too - skipping logdump for '${node}'"
    fi
}

# Save logs for node $1 into directory $2. Pass in any non-common files in $3.
# Pass in any non-common systemd services in $4.
# $3 and $4 should be a space-separated list of files.
# Set $5 to true to indicate it is on master. Default to false.
# This function shouldn't ever trigger errexit
function save-logs() {
    local -r node_name="${1}"
    local -r dir="${2}"
    local files=()
    IFS=' ' read -r -a files <<< "$3"
    local opt_systemd_services="${4:-""}"
    local on_master="${5:-"false"}"

    local extra=()
    IFS=' ' read -r -a extra <<< "$extra_log_files"
    files+=("${extra[@]}")
    if [[ -n "${use_custom_instance_list}" ]]; then
      if [[ -n "${LOG_DUMP_SAVE_LOGS:-}" ]]; then
        local dump=()
        IFS=' ' read -r -a dump <<< "${LOG_DUMP_SAVE_LOGS:-}"
        files+=("${dump[@]}")
      fi
    else
      local providerlogs=()
      case "${KUBERNETES_PROVIDER}" in
        gce|gke)
          IFS=' ' read -r -a providerlogs <<< "${gce_logfiles}"
          ;;
        aws)
          IFS=' ' read -r -a providerlogs <<< "${aws_logfiles}"
          ;;
      esac
      files+=("${providerlogs[@]}")
    fi
    local services
    read -r -a services <<< "${systemd_services} ${opt_systemd_services} ${extra_systemd_services}"

    if log-dump-ssh "${node_name}" "command -v journalctl" &> /dev/null; then
        if [[ "${on_master}" == "true" ]]; then
          log-dump-ssh "${node_name}" "sudo journalctl --output=short-precise -u kube-master-installation.service" > "${dir}/kube-master-installation.log" || true
          log-dump-ssh "${node_name}" "sudo journalctl --output=short-precise -u kube-master-configuration.service" > "${dir}/kube-master-configuration.log" || true
        else
          log-dump-ssh "${node_name}" "sudo journalctl --output=short-precise -u kube-node-installation.service" > "${dir}/kube-node-installation.log" || true
          log-dump-ssh "${node_name}" "sudo journalctl --output=short-precise -u kube-node-configuration.service" > "${dir}/kube-node-configuration.log" || true
        fi
        log-dump-ssh "${node_name}" "sudo journalctl --output=short-precise -k" > "${dir}/kern.log" || true

        for svc in "${services[@]}"; do
            log-dump-ssh "${node_name}" "sudo journalctl --output=short-precise -u ${svc}.service" > "${dir}/${svc}.log" || true
        done

        if [[ "$dump_systemd_journal" == "true" ]]; then
          log-dump-ssh "${node_name}" "sudo journalctl --output=short-precise" > "${dir}/systemd.log" || true
        fi
    else
        local tmpfiles=()
        for f in "${kern_logfile}" "${initd_logfiles}" "${supervisord_logfiles}"; do
            IFS=' ' read -r -a tmpfiles <<< "$f"
            files+=("${tmpfiles[@]}")
        done
    fi

    # Try dumping coverage profiles, if it looks like coverage is enabled in the first place.
    if log-dump-ssh "${node_name}" "stat /var/log/kubelet.cov" &> /dev/null; then
      if log-dump-ssh "${node_name}" "command -v docker" &> /dev/null; then
        if [[ "${on_master}" == "true" ]]; then
          run-in-docker-container "${node_name}" "kube-apiserver" "cat /tmp/k8s-kube-apiserver.cov" > "${dir}/kube-apiserver.cov" || true
          run-in-docker-container "${node_name}" "kube-scheduler" "cat /tmp/k8s-kube-scheduler.cov" > "${dir}/kube-scheduler.cov" || true
          run-in-docker-container "${node_name}" "kube-controller-manager" "cat /tmp/k8s-kube-controller-manager.cov" > "${dir}/kube-controller-manager.cov" || true
        else
          run-in-docker-container "${node_name}" "kube-proxy" "cat /tmp/k8s-kube-proxy.cov" > "${dir}/kube-proxy.cov" || true
        fi
      else
        echo 'Coverage profiles seem to exist, but cannot be retrieved from inside containers.'
      fi
    fi

    echo 'Changing logfiles to be world-readable for download'
    log-dump-ssh "${node_name}" "sudo chmod -R a+r /var/log" || true

    echo "Copying '${files[*]}' from ${node_name}"
    copy-logs-from-node "${node_name}" "${dir}" "${files[@]}"
}

# Saves a copy of the Windows Docker event log to ${WINDOWS_LOGS_DIR}\docker.log
# on node $1.
function export-windows-docker-event-log() {
    local -r node="${1}"

    local -r powershell_cmd="powershell.exe -Command \"\$logs=\$(Get-EventLog -LogName Application -Source Docker | Format-Table -Property TimeGenerated, EntryType, Message -Wrap); \$logs | Out-File -FilePath '${WINDOWS_LOGS_DIR}\\docker.log'\""

    # Retry up to 3 times to allow ssh keys to be properly propagated and
    # stored.
    for retry in {1..3}; do
      if gcloud compute ssh --project "${PROJECT}" --zone "${ZONE}" "${node}" \
        --command "$powershell_cmd"; then
        break
      else
        sleep 10
      fi
    done
}

# Saves prepulled Windows Docker images list to ${WINDOWS_LOGS_DIR}\docker_images.log
# on node $1.
function export-windows-docker-images-list() {
    local -r node="${1}"

    local -r powershell_cmd="powershell.exe -Command \"\$logs=\$(docker image list); \$logs | Out-File -FilePath '${WINDOWS_LOGS_DIR}\\docker_images.log'\""

    # Retry up to 3 times to allow ssh keys to be properly propagated and
    # stored.
    for retry in {1..3}; do
      if gcloud compute ssh --project "${PROJECT}" --zone "${ZONE}" "${node}" \
        --command "$powershell_cmd"; then
        break
      else
        sleep 10
      fi
    done
}

# Saves log files from diagnostics tool.(https://github.com/GoogleCloudPlatform/compute-image-tools/tree/master/cli_tools/diagnostics)
function save-windows-logs-via-diagnostics-tool() {
    local node="${1}"
    local dest_dir="${2}"

    gcloud compute instances add-metadata "${node}" --metadata enable-diagnostics=true --project="${PROJECT}" --zone="${ZONE}"
    local logs_archive_in_gcs
    logs_archive_in_gcs=$(gcloud alpha compute diagnose export-logs "${node}" "--zone=${ZONE}" "--project=${PROJECT}" | tail -n 1)
    local temp_local_path="${node}.zip"
    for retry in {1..20}; do
      if gsutil mv "${logs_archive_in_gcs}" "${temp_local_path}"  > /dev/null 2>&1; then
        echo "Downloaded diagnostics log from ${logs_archive_in_gcs}"
        break
      else
        sleep 10
      fi
    done

    if [[ -f "${temp_local_path}" ]]; then
      unzip "${temp_local_path}" -d "${dest_dir}" > /dev/null
      rm -f "${temp_local_path}"
    fi
}

# Saves log files from SSH
function save-windows-logs-via-ssh() {
    local node="${1}"
    local dest_dir="${2}"

    export-windows-docker-event-log "${node}"
    export-windows-docker-images-list "${node}"

    local remote_files=()
    for file in "${windows_node_logfiles[@]}"; do
      remote_files+=( "${WINDOWS_LOGS_DIR}\\${file}" )
    done
    remote_files+=( "${windows_node_otherfiles[@]}" )

    # TODO(pjh, yujuhong): handle rotated logs and copying multiple files at the
    # same time.
    for remote_file in "${remote_files[@]}"; do
      # Retry up to 3 times to allow ssh keys to be properly propagated and
      # stored.
      for retry in {1..3}; do
        if gcloud compute scp --recurse --project "${PROJECT}" \
          --zone "${ZONE}" "${node}:${remote_file}" "${dest_dir}" \
          > /dev/null; then
          break
        else
          sleep 10
        fi
      done
    done
}

# Save log files and serial console output from Windows node $1 into local
# directory $2.
# This function shouldn't ever trigger errexit.
function save-logs-windows() {
    local -r node="${1}"
    local -r dest_dir="${2}"

    if [[ ! "${gcloud_supported_providers}" =~ ${KUBERNETES_PROVIDER} ]]; then
      echo "Not saving logs for ${node}, Windows log dumping requires gcloud support"
      return
    fi

    if [[ "${KUBERNETES_PROVIDER}" == "gke" ]]; then
      save-windows-logs-via-diagnostics-tool "${node}" "${dest_dir}"
    else
      save-windows-logs-via-ssh "${node}" "${dest_dir}"
    fi

    # Serial port 1 contains the Windows console output.
    gcloud compute instances get-serial-port-output --project "${PROJECT}" \
      --zone "${ZONE}" --port 1 "${node}" > "${dest_dir}/serial-1.log" || true
}

# Execute a command in container $2 on node $1.
# Uses docker because the container may not ordinarily permit direct execution.
function run-in-docker-container() {
  local node_name="$1"
  local container="$2"
  shift 2
  log-dump-ssh "${node_name}" "docker exec \"\$(docker ps -f label=io.kubernetes.container.name=${container} --format \"{{.ID}}\")\" $*"
}

function dump_masters() {
  local master_names=()
  if [[ -n "${use_custom_instance_list}" ]]; then
    while IFS='' read -r line; do master_names+=("$line"); done < <(log_dump_custom_get_instances master)
  elif [[ ! "${master_ssh_supported_providers}" =~ ${KUBERNETES_PROVIDER} ]]; then
    echo "Master SSH not supported for ${KUBERNETES_PROVIDER}"
    return
  elif [[ -n "${KUBEMARK_MASTER_NAME:-}" ]]; then
    master_names=( "${KUBEMARK_MASTER_NAME}" )
  else
    if ! (detect-master); then
      echo 'Master not detected. Is the cluster up?'
      return
    fi
    master_names=( "${MASTER_NAME}" )
  fi

  if [[ "${#master_names[@]}" == 0 ]]; then
    echo 'No masters found?'
    return
  fi

  proc=${max_dump_processes}
  for master_name in "${master_names[@]}"; do
    master_dir="${report_dir}/${master_name}"
    mkdir -p "${master_dir}"
    save-logs "${master_name}" "${master_dir}" "${master_logfiles}" "" "true" &

    # We don't want to run more than ${max_dump_processes} at a time, so
    # wait once we hit that many nodes. This isn't ideal, since one might
    # take much longer than the others, but it should help.
    proc=$((proc - 1))
    if [[ proc -eq 0 ]]; then
      proc=${max_dump_processes}
      wait
    fi
  done
  # Wait for any remaining processes.
  if [[ proc -gt 0 && proc -lt ${max_dump_processes} ]]; then
    wait
  fi
}

# Dumps logs from nodes in the cluster. Linux nodes to dump logs from can be
# specified via $1 or $use_custom_instance_list. If not specified then the nodes
# to dump logs for will be detected using detect-node-names(); if Windows nodes
# are present then they will be detected and their logs will be dumped too.
function dump_nodes() {
  local node_names=()
  local windows_node_names=()
  if [[ -n "${1:-}" ]]; then
    echo 'Dumping logs for nodes provided as args to dump_nodes() function'
    node_names=( "$@" )
  else
    echo 'Detecting nodes in the cluster'
    detect-node-names &> /dev/null
    if [[ -n "${NODE_NAMES:-}" ]]; then
      node_names=( "${NODE_NAMES[@]}" )
    fi
    if [[ -n "${WINDOWS_NODE_NAMES:-}" ]]; then
      windows_node_names=( "${WINDOWS_NODE_NAMES[@]}" )
    fi
  fi

  if [[ "${#node_names[@]}" == 0 && "${#windows_node_names[@]}" == 0 ]]; then
    echo 'No nodes found!'
    return
  fi

  node_logfiles_all="${node_logfiles}"
  if [[ "${ENABLE_HOLLOW_NODE_LOGS:-}" == "true" ]]; then
    node_logfiles_all="${node_logfiles_all} ${hollow_node_logfiles}"
  fi

  linux_nodes_selected_for_logs=()
  if [[ -n "${LOGDUMP_ONLY_N_RANDOM_NODES:-}" ]]; then
    # We randomly choose 'LOGDUMP_ONLY_N_RANDOM_NODES' many nodes for fetching logs.
    for index in $(shuf -i 0-$(( ${#node_names[*]} - 1 )) -n "${LOGDUMP_ONLY_N_RANDOM_NODES}")
    do
      linux_nodes_selected_for_logs+=("${node_names[$index]}")
    done
  else
    linux_nodes_selected_for_logs=( "${node_names[@]}" )
  fi
  all_selected_nodes=( "${linux_nodes_selected_for_logs[@]}" )
  all_selected_nodes+=( "${windows_node_names[@]}" )

  proc=${max_dump_processes}
  start="$(date +%s)"
  # log_dump_ssh_timeout is the maximal number of seconds the log dumping over
  # SSH operation can take. Please note that the logic enforcing the timeout
  # is only a best effort. The actual time of the operation may be longer
  # due to waiting for all the child processes below.
  log_dump_ssh_timeout_seconds="${LOG_DUMP_SSH_TIMEOUT_SECONDS:-}"
  for i in "${!all_selected_nodes[@]}"; do
    node_name="${all_selected_nodes[$i]}"
    node_dir="${report_dir}/${node_name}"
    mkdir -p "${node_dir}"
    if [[ "${i}" -lt "${#linux_nodes_selected_for_logs[@]}" ]]; then
      # Save logs in the background. This speeds up things when there are
      # many nodes.
      save-logs "${node_name}" "${node_dir}" "${node_logfiles_all}" "${node_systemd_services}" &
    else
      save-logs-windows "${node_name}" "${node_dir}" &
    fi

    # We don't want to run more than ${max_dump_processes} at a time, so
    # wait once we hit that many nodes. This isn't ideal, since one might
    # take much longer than the others, but it should help.
    proc=$((proc - 1))
    if [[ proc -eq 0 ]]; then
      proc=${max_dump_processes}
      wait
      now="$(date +%s)"
      if [[ -n "${log_dump_ssh_timeout_seconds}" && $((now - start)) -gt ${log_dump_ssh_timeout_seconds} ]]; then
        echo "WARNING: Hit timeout after ${log_dump_ssh_timeout_seconds} seconds, finishing log dumping over SSH shortly"
        break
      fi
    fi
  done
  # Wait for any remaining processes.
  if [[ proc -gt 0 && proc -lt ${max_dump_processes} ]]; then
    wait
  fi
}

# Collect names of nodes which didn't run logexporter successfully.
# This function examines NODE_NAMES but not WINDOWS_NODE_NAMES since logexporter
# does not run on Windows nodes.
#
# Note: This step is O(#nodes^2) as we check if each node is present in the list of succeeded nodes.
# Making it linear would add code complexity without much benefit (as it just takes ~1s for 5k nodes).
# Assumes:
#   NODE_NAMES
# Sets:
#   NON_LOGEXPORTED_NODES
function find_non_logexported_nodes() {
  local file="${gcs_artifacts_dir}/logexported-nodes-registry"
  echo "Listing marker files ($file) for successful nodes..."
  succeeded_nodes=$(gsutil ls "${file}") || return 1
  echo 'Successfully listed marker files for successful nodes'
  NON_LOGEXPORTED_NODES=()
  for node in "${NODE_NAMES[@]}"; do
    if [[ ! "${succeeded_nodes}" =~ ${node} ]]; then
      NON_LOGEXPORTED_NODES+=("${node}")
    fi
  done
}

# This function examines NODE_NAMES but not WINDOWS_NODE_NAMES since logexporter
# does not run on Windows nodes.
function dump_nodes_with_logexporter() {
  detect-node-names &> /dev/null

  if [[ -z "${NODE_NAMES:-}" ]]; then
    echo 'No nodes found!'
    return
  fi

  # Obtain parameters required by logexporter.
  if [[ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
    local -r service_account_credentials="$(base64 "${GOOGLE_APPLICATION_CREDENTIALS}" | tr -d '\n')"
  fi
  local -r cloud_provider="${KUBERNETES_PROVIDER}"
  local -r enable_hollow_node_logs="${ENABLE_HOLLOW_NODE_LOGS:-false}"
  local -r logexport_sleep_seconds="$(( 90 + NUM_NODES / 3 ))"
  if [[ -z "${ZONE_NODE_SELECTOR_DISABLED:-}" ]]; then
    local -r node_selector="${ZONE_NODE_SELECTOR_LABEL:-topology.kubernetes.io/zone}: ${ZONE}"
  fi
  local -r use_application_default_credentials="${LOGEXPORTER_USE_APPLICATION_DEFAULT_CREDENTIALS:-false}"

  # Fill in the parameters in the logexporter daemonset template.
  local -r tmp="${KUBE_TEMP}/logexporter"
  local -r manifest_yaml="${tmp}/logexporter-daemonset.yaml"
  mkdir -p "${tmp}"
  local -r cwd=$(dirname "${BASH_SOURCE[0]}")
  cp "${cwd}/logexporter-daemonset.yaml" "${manifest_yaml}"

  sed -i'' -e "s@{{.NodeSelector}}@${node_selector:-}@g" "${manifest_yaml}"
  sed -i'' -e "s@{{.LogexporterNamespace}}@${logexporter_namespace}@g" "${manifest_yaml}"
  sed -i'' -e "s@{{.ServiceAccountCredentials}}@${service_account_credentials:-}@g" "${manifest_yaml}"
  sed -i'' -e "s@{{.CloudProvider}}@${cloud_provider}@g" "${manifest_yaml}"
  sed -i'' -e "s@{{.GCSPath}}@${gcs_artifacts_dir}@g" "${manifest_yaml}"
  sed -i'' -e "s@{{.EnableHollowNodeLogs}}@${enable_hollow_node_logs}@g" "${manifest_yaml}"
  sed -i'' -e "s@{{.DumpSystemdJournal}}@${dump_systemd_journal}@g" "${manifest_yaml}"
  sed -i'' -e "s@{{.ExtraLogFiles}}@${extra_log_files}@g" "${manifest_yaml}"
  sed -i'' -e "s@{{.ExtraSystemdServices}}@${extra_systemd_services}@g" "${manifest_yaml}"
  sed -i'' -e "s@{{.UseApplicationDefaultCredentials}}@${use_application_default_credentials}@g" "${manifest_yaml}"

  # Create the logexporter namespace, service-account secret and the logexporter daemonset within that namespace.
  if ! kubectl create -f "${manifest_yaml}"; then
    echo 'Failed to create logexporter daemonset.. falling back to logdump through SSH'
    kubectl delete namespace "${logexporter_namespace}" || true
    dump_nodes "${NODE_NAMES[@]}"
    logexporter_failed=1
    return
  fi

  # Periodically fetch list of already logexported nodes to verify
  # if we aren't already done.
  start="$(date +%s)"
  while true; do
    now="$(date +%s)"
    if [[ $((now - start)) -gt ${logexport_sleep_seconds} ]]; then
      echo 'Waiting for all nodes to be logexported timed out.'
      break
    fi
    if find_non_logexported_nodes; then
      if [[ -z "${NON_LOGEXPORTED_NODES:-}" ]]; then
        break
      fi
    fi
    sleep 15
  done

  # Store logs from logexporter pods to allow debugging log exporting process
  # itself.
  proc=${max_dump_processes}
  kubectl get pods -n "${logexporter_namespace}" -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.nodeName}{"\n"}{end}' | (while read -r pod node; do
    echo "Fetching logs from ${pod} running on ${node}"
    mkdir -p "${report_dir}/${node}"
    kubectl logs -n "${logexporter_namespace}" "${pod}" > "${report_dir}/${node}/${pod}.log" &

    # We don't want to run more than ${max_dump_processes} at a time, so
    # wait once we hit that many nodes. This isn't ideal, since one might
    # take much longer than the others, but it should help.
    proc=$((proc - 1))
    if [[ proc -eq 0 ]]; then
      proc=${max_dump_processes}
      wait
    fi
  # Wait for any remaining processes.
  done; wait)

  # List registry of marker files (of nodes whose logexporter succeeded) from GCS.
  for retry in {1..10}; do
    if find_non_logexported_nodes; then
      break
    else
      echo "Attempt ${retry} failed to list marker files for successful nodes"
      if [[ "${retry}" == 10 ]]; then
        echo 'Final attempt to list marker files failed.. falling back to logdump through SSH'
        # Timeout prevents the test waiting too long to delete resources and
        # never uploading logs, as happened in https://github.com/kubernetes/kubernetes/issues/111111
        kubectl delete namespace "${logexporter_namespace}" --timeout 15m || true
        dump_nodes "${NODE_NAMES[@]}"
        logexporter_failed=1
        return
      fi
      sleep 2
    fi
  done

  failed_nodes=()
  # The following if is needed, because defaulting for empty arrays
  # seems to treat them as non-empty with single empty string.
  if [[ -n "${NON_LOGEXPORTED_NODES:-}" ]]; then
    for node in "${NON_LOGEXPORTED_NODES[@]:-}"; do
      echo "Logexporter didn't succeed on node ${node}. Queuing it for logdump through SSH."
      failed_nodes+=("${node}")
    done
  fi

  # If less than a certain ratio of the nodes got logexported, report an error.
  if [[ $(((${#NODE_NAMES[@]} - ${#failed_nodes[@]}) * 100)) -lt $((${#NODE_NAMES[@]} * log_dump_expected_success_percentage )) ]]; then
    logexporter_failed=1
  fi

  # Delete the logexporter resources and dump logs for the failed nodes (if any) through SSH.
  kubectl get pods --namespace "${logexporter_namespace}" || true
  # Timeout prevents the test waiting too long to delete resources and
  # never uploading logs, as happened in https://github.com/kubernetes/kubernetes/issues/111111
  kubectl delete namespace "${logexporter_namespace}" --timeout 15m || true
  if [[ "${#failed_nodes[@]}" != 0 ]]; then
    echo -e "Dumping logs through SSH for the following nodes:\n${failed_nodes[*]}"
    dump_nodes "${failed_nodes[@]}"
  fi
}

# Writes node information that's available through the gcloud and kubectl API
# surfaces to a nodes/ subdirectory of $report_dir.
function dump_node_info() {
  if [[ "${SKIP_DUMP_NODE_INFO:-}" == "true" ]]; then
    echo 'Skipping dumping of node info'
    return
  fi

  nodes_dir="${report_dir}/nodes"
  mkdir -p "${nodes_dir}"

  detect-node-names
  if [[ -n "${NODE_NAMES:-}" ]]; then
    printf "%s\n" "${NODE_NAMES[@]}" > "${nodes_dir}/node_names.txt"
  fi
  if [[ -n "${WINDOWS_NODE_NAMES:-}" ]]; then
    printf "%s\n" "${WINDOWS_NODE_NAMES[@]}" > "${nodes_dir}/windows_node_names.txt"
  fi

  # If we are not able to reach the server, just bail out as the other
  # kubectl calls below will fail anyway (we don't want to error out collecting logs)
  kubectl version || return 0

  kubectl get nodes -o yaml > "${nodes_dir}/nodes.yaml"

  api_node_names=()
  api_node_names+=($( kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\tReady="}{@.status.conditions[?(@.type=="Ready")].status}{"\n"}{end}' | awk '/Ready=True/ {print $1}'))
  if [[ "${#api_node_names[@]}" -le 5 ]]; then
    for node_name in "${api_node_names[@]}"; do
      mkdir -p "${nodes_dir}/${node_name}"
      kubectl get --raw "/api/v1/nodes/${node_name}/proxy/metrics" > "${nodes_dir}/${node_name}/kubelet_metrics.txt"
    done
  fi
}

function detect_node_failures() {
  if ! [[ "${gcloud_supported_providers}" =~ ${KUBERNETES_PROVIDER} ]]; then
    return
  fi

  detect-node-names
  if [[ "${KUBERNETES_PROVIDER}" == "gce" ]]; then
    local all_instance_groups=("${INSTANCE_GROUPS[@]}" "${WINDOWS_INSTANCE_GROUPS[@]}")
  else
    local all_instance_groups=("${INSTANCE_GROUPS[@]}")
  fi

  if [ -z "${all_instance_groups:-}" ]; then
    return
  fi
  for group in "${all_instance_groups[@]}"; do
    local creation_timestamp
    creation_timestamp=$(gcloud compute instance-groups managed describe \
                         "${group}" \
                         --project "${PROJECT}" \
                         --zone "${ZONE}" \
                         --format='value(creationTimestamp)')
    echo "Failures for ${group} (if any):"
    gcloud logging read --order=asc \
          --format='table(timestamp,jsonPayload.resource.name,jsonPayload.event_subtype)' \
          --project "${PROJECT}" \
          "resource.type=\"gce_instance\"
           logName=\"projects/${PROJECT}/logs/compute.googleapis.com%2Factivity_log\"
           (jsonPayload.event_subtype=\"compute.instances.hostError\" OR jsonPayload.event_subtype=\"compute.instances.automaticRestart\")
           jsonPayload.resource.name:\"${group}\"
           timestamp >= \"${creation_timestamp}\""
  done
}

function dump_logs() {
  # Copy master logs to artifacts dir locally (through SSH).
  echo "Dumping logs from master locally to '${report_dir}'"
  dump_masters
  if [[ "${DUMP_ONLY_MASTER_LOGS:-}" == "true" ]]; then
    echo 'Skipping dumping of node logs'
    return
  fi

  # Copy logs from nodes to GCS directly or to artifacts dir locally (through SSH).
  if [[ -n "${gcs_artifacts_dir}" ]]; then
    echo "Dumping logs from nodes to GCS directly at '${gcs_artifacts_dir}' using logexporter"
    dump_nodes_with_logexporter
  else
    echo "Dumping logs from nodes locally to '${report_dir}'"
    dump_nodes
  fi
}

# Without ${DUMP_TO_GCS_ONLY} == true:
# * only logs exported by logexporter will be uploaded to
#   ${gcs_artifacts_dir}
# * other logs (master logs, nodes where logexporter failed) will be
#   fetched locally to ${report_dir}.
# If $DUMP_TO_GCS_ONLY == 'true', all logs will be uploaded directly to
# ${gcs_artifacts_dir}.
function main() {
  setup
  kube::util::ensure-temp-dir
  if [[ "${DUMP_TO_GCS_ONLY:-}" == "true" ]] && [[ -n "${gcs_artifacts_dir}" ]]; then
    report_dir="${KUBE_TEMP}/logs"
    mkdir -p "${report_dir}"
    echo "${gcs_artifacts_dir}" > "${local_report_dir}/master-and-node-logs.link.txt"
    echo "Dumping logs temporarily to '${report_dir}'. Will upload to '${gcs_artifacts_dir}' later."
  else
    report_dir="${local_report_dir}"
  fi

  dump_logs
  dump_node_info

  if [[ "${DUMP_TO_GCS_ONLY:-}" == "true" ]] && [[ -n "${gcs_artifacts_dir}" ]]; then
    if [[ "$(ls -A ${report_dir})" ]]; then
      echo "Uploading '${report_dir}' to '${gcs_artifacts_dir}'"

      if gsutil ls "${gcs_artifacts_dir}" > /dev/null; then
        # If "${gcs_artifacts_dir}" exists, the simple call:
        # `gsutil cp -r /tmp/dir/logs ${gcs_artifacts_dir}` will
        #  create subdirectory 'logs' in ${gcs_artifacts_dir}
        #
        # If "${gcs_artifacts_dir}" exists, we want to merge its content
        # with local logs. To do that we do the following trick:
        # * Let's say that ${gcs_artifacts_dir} == 'gs://a/b/c'.
        # * We rename 'logs' to 'c'
        # * Call `gsutil cp -r /tmp/dir/c gs://a/b/`
        #
        # Similar pattern is used in bootstrap.py#L409-L416.
        # It is a known issue that gsutil cp behavior is that complex.
        # For more information on this, see:
        # https://cloud.google.com/storage/docs/gsutil/commands/cp#how-names-are-constructed
        remote_dir=$(dirname "${gcs_artifacts_dir}")
        remote_basename=$(basename "${gcs_artifacts_dir}")
        mv "${report_dir}" "${KUBE_TEMP}/${remote_basename}"
        gsutil -m cp -r -c -z log,txt,xml "${KUBE_TEMP}/${remote_basename}" "${remote_dir}"
        rm -rf "${KUBE_TEMP}/${remote_basename:?}"
      else  # ${gcs_artifacts_dir} doesn't exist.
        gsutil -m cp -r -c -z log,txt,xml "${report_dir}" "${gcs_artifacts_dir}"
        rm -rf "${report_dir}"
      fi
    else
      echo "Skipping upload of '${report_dir}' as it's empty."
    fi
  fi

  detect_node_failures
  if [[ ${logexporter_failed} -ne 0 && ${log_dump_expected_success_percentage} -gt 0 ]]; then
    return 1
  fi
}

main
