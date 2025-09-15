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

# TODO(shyamjvs): This script should be moved to test/e2e which is where it ideally belongs.
set -o errexit
set -o nounset
set -o pipefail

readonly report_dir="${1:-_artifacts}"
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
readonly node_ssh_supported_providers="gce gke aws"
readonly gcloud_supported_providers="gce gke"

readonly master_logfiles="kube-apiserver.log kube-apiserver-audit.log kube-scheduler.log kube-controller-manager.log cloud-controller-manager.log etcd.log etcd-events.log glbc.log cluster-autoscaler.log kube-addon-manager.log konnectivity-server.log fluentd.log kubelet.cov"
readonly node_logfiles="kube-proxy.log containers/konnectivity-agent-*.log fluentd.log node-problem-detector.log kubelet.cov kube-network-policies.log"
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

function print-deprecation-note() {
  local -r dashline=$(printf -- '-%.0s' {1..100})
  echo "${dashline}"
  echo "k/k version of the log-dump.sh script is deprecated!"
  echo "Please migrate your test job to use test-infra's repo version of log-dump.sh!"
  echo "Migration steps can be found in the readme file."
  echo "${dashline}"
}

# TODO: Get rid of all the sourcing of bash dependencies eventually.
function setup() {
  KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/../..
  if [[ -z "${use_custom_instance_list}" ]]; then
    : "${KUBE_CONFIG_FILE:=config-test.sh}"
    echo 'Sourcing kube-util.sh'
    source "${KUBE_ROOT}/cluster/kube-util.sh"
    echo 'Detecting project'
    detect-project 2>&1
  elif [[ "${KUBERNETES_PROVIDER}" == "gke" ]]; then
    echo "Using 'use_custom_instance_list' with gke, skipping check for LOG_DUMP_SSH_KEY and LOG_DUMP_SSH_USER"
    # Source the below script for the ssh-to-node utility function.
    # Hack to save and restore the value of the ZONE env as the script overwrites it.
    local gke_zone="${ZONE:-}"
    source "${KUBE_ROOT}/cluster/gce/util.sh"
    ZONE="${gke_zone}"
  elif [[ -z "${LOG_DUMP_SSH_KEY:-}" ]]; then
    echo 'LOG_DUMP_SSH_KEY not set, but required when using log_dump_custom_get_instances'
    exit 1
  elif [[ -z "${LOG_DUMP_SSH_USER:-}" ]]; then
    echo 'LOG_DUMP_SSH_USER not set, but required when using log_dump_custom_get_instances'
    exit 1
  fi
  source "${KUBE_ROOT}/hack/lib/util.sh"
}

function log-dump-ssh() {
  if [[ "${gcloud_supported_providers}" =~ ${KUBERNETES_PROVIDER} ]]; then
    ssh-to-node "$@"
    return
  fi

  local host="$1"
  local cmd="$2"

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
      source_file_args=()
      for single_file in "${files[@]}"; do
        source_file_args+=( "${node}:${single_file}" )
      done
      gcloud compute scp --recurse --project "${PROJECT}" --zone "${ZONE}" "${source_file_args[@]}" "${dir}" > /dev/null || true
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

    # log where we pull the images from
    log-dump-ssh "${node_name}" "sudo ctr -n k8s.io images ls" > "${dir}/images-containerd.log" || true
    log-dump-ssh "${node_name}" "sudo docker images --all" > "${dir}/images-docker.log" || true

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
  elif [[ -n "${use_custom_instance_list}" ]]; then
    echo 'Dumping logs for nodes provided by log_dump_custom_get_instances() function'
    while IFS='' read -r line; do node_names+=("$line"); done < <(log_dump_custom_get_instances node)
  elif [[ ! "${node_ssh_supported_providers}" =~ ${KUBERNETES_PROVIDER} ]]; then
    echo "Node SSH not supported for ${KUBERNETES_PROVIDER}"
    return
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
  if [[ -n "${use_custom_instance_list}" ]]; then
    echo 'Dumping logs for nodes provided by log_dump_custom_get_instances() function'
    NODE_NAMES=()
    while IFS='' read -r line; do NODE_NAMES+=("$line"); done < <(log_dump_custom_get_instances node)
  else
    echo 'Detecting nodes in the cluster'
    detect-node-names &> /dev/null
  fi

  if [[ -z "${NODE_NAMES:-}" ]]; then
    echo 'No nodes found!'
    return
  fi

  # Obtain parameters required by logexporter.
  local -r service_account_credentials="$(base64 "${GOOGLE_APPLICATION_CREDENTIALS}" | tr -d '\n')"
  local -r cloud_provider="${KUBERNETES_PROVIDER}"
  local -r enable_hollow_node_logs="${ENABLE_HOLLOW_NODE_LOGS:-false}"
  local -r logexport_sleep_seconds="$(( 90 + NUM_NODES / 3 ))"
  if [[ -z "${ZONE_NODE_SELECTOR_DISABLED:-}" ]]; then
    local -r node_selector="${ZONE_NODE_SELECTOR_LABEL:-topology.kubernetes.io/zone}: ${ZONE}"
  fi

  # Fill in the parameters in the logexporter daemonset template.
  local -r tmp="${KUBE_TEMP}/logexporter"
  local -r manifest_yaml="${tmp}/logexporter-daemonset.yaml"
  mkdir -p "${tmp}"
  cp "${KUBE_ROOT}/cluster/log-dump/logexporter-daemonset.yaml" "${manifest_yaml}"

  sed -i'' -e "s@{{.NodeSelector}}@${node_selector:-}@g" "${manifest_yaml}"
  sed -i'' -e "s@{{.LogexporterNamespace}}@${logexporter_namespace}@g" "${manifest_yaml}"
  sed -i'' -e "s@{{.ServiceAccountCredentials}}@${service_account_credentials}@g" "${manifest_yaml}"
  sed -i'' -e "s@{{.CloudProvider}}@${cloud_provider}@g" "${manifest_yaml}"
  sed -i'' -e "s@{{.GCSPath}}@${gcs_artifacts_dir}@g" "${manifest_yaml}"
  sed -i'' -e "s@{{.EnableHollowNodeLogs}}@${enable_hollow_node_logs}@g" "${manifest_yaml}"
  sed -i'' -e "s@{{.DumpSystemdJournal}}@${dump_systemd_journal}@g" "${manifest_yaml}"
  sed -i'' -e "s@{{.ExtraLogFiles}}@${extra_log_files}@g" "${manifest_yaml}"
  sed -i'' -e "s@{{.ExtraSystemdServices}}@${extra_systemd_services}@g" "${manifest_yaml}"

  # Create the logexporter namespace, service-account secret and the logexporter daemonset within that namespace.
  KUBECTL="${KUBE_ROOT}/cluster/kubectl.sh"
  if ! "${KUBECTL}" create -f "${manifest_yaml}"; then
    echo 'Failed to create logexporter daemonset.. falling back to logdump through SSH'
    "${KUBECTL}" delete namespace "${logexporter_namespace}" || true
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
  "${KUBECTL}" get pods -n "${logexporter_namespace}" -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.nodeName}{"\n"}{end}' | (while read -r pod node; do
    echo "Fetching logs from ${pod} running on ${node}"
    mkdir -p "${report_dir}/${node}"
    "${KUBECTL}" logs -n "${logexporter_namespace}" "${pod}" > "${report_dir}/${node}/${pod}.log" &

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
        "${KUBECTL}" delete namespace "${logexporter_namespace}" || true
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
  "${KUBECTL}" get pods --namespace "${logexporter_namespace}" || true
  "${KUBECTL}" delete namespace "${logexporter_namespace}" || true
  if [[ "${#failed_nodes[@]}" != 0 ]]; then
    echo -e "Dumping logs through SSH for the following nodes:\n${failed_nodes[*]}"
    dump_nodes "${failed_nodes[@]}"
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

function main() {
  print-deprecation-note
  setup
  kube::util::ensure-temp-dir
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

  detect_node_failures
  if [[ ${logexporter_failed} -ne 0 && ${log_dump_expected_success_percentage} -gt 0 ]]; then
    return 1
  fi
}

main
