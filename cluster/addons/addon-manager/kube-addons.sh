#!/usr/bin/env bash

# Copyright 2014 The Kubernetes Authors.
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

# LIMITATIONS
# 1. Exit code is probably not always correct.
# 2. There are no unittests.
# 3. Will not work if the total length of paths to addons is greater than
#    bash can handle. Probably it is not a problem: ARG_MAX=2097152 on GCE.

# cosmetic improvements to be done
# 1. Improve the log function; add timestamp, file name, etc.
# 2. Logging doesn't work from files that print things out.
# 3. Kubectl prints the output to stderr (the output should be captured and then
#    logged)

KUBECTL=${KUBECTL_BIN:-/usr/local/bin/kubectl}
KUBECTL_OPTS=${KUBECTL_OPTS:-}
# KUBECTL_PRUNE_WHITELIST is a list of resources whitelisted by default.
# This is currently the same with the default in:
# https://github.com/kubernetes/kubectl/blob/master/pkg/cmd/apply/prune.go.
# To override the default list with other values, set
# KUBECTL_PRUNE_WHITELIST_OVERRIDE environment variable to space-separated
# names of resources to whitelist.
if [ -z "${KUBECTL_PRUNE_WHITELIST_OVERRIDE:-}" ]; then
  KUBECTL_PRUNE_WHITELIST=(
    core/v1/ConfigMap
    core/v1/Endpoints
    core/v1/Namespace
    core/v1/PersistentVolumeClaim
    core/v1/PersistentVolume
    core/v1/Pod
    core/v1/ReplicationController
    core/v1/Secret
    core/v1/Service
    batch/v1/Job
    batch/v1/CronJob
    apps/v1/DaemonSet
    apps/v1/Deployment
    apps/v1/ReplicaSet
    apps/v1/StatefulSet
    networking.k8s.io/v1/Ingress
  )
else
  read -ra KUBECTL_PRUNE_WHITELIST <<< "${KUBECTL_PRUNE_WHITELIST_OVERRIDE}"
fi

# This variable is unused in this file, but not in those that source it.
# shellcheck disable=SC2034
ADDON_CHECK_INTERVAL_SEC=${TEST_ADDON_CHECK_INTERVAL_SEC:-60}
ADDON_PATH=${ADDON_PATH:-/etc/kubernetes/addons}

# This variable is unused in this file, but not in those that source it.
# shellcheck disable=SC2034
SYSTEM_NAMESPACE=kube-system

# Addons could use this label with two modes:
# - ADDON_MANAGER_LABEL=Reconcile
# - ADDON_MANAGER_LABEL=EnsureExists
ADDON_MANAGER_LABEL="addonmanager.kubernetes.io/mode"
# This label is deprecated (only for Addon Manager). In future release
# addon-manager may not respect it anymore. Addons with
# CLUSTER_SERVICE_LABEL=true and without ADDON_MANAGER_LABEL=EnsureExists
# will be reconciled for now.
CLUSTER_SERVICE_LABEL="kubernetes.io/cluster-service"

# Whether only one addon manager should be running in a multi-master setup.
# Disabling this flag will force all addon managers to assume they are the
# leaders.
ADDON_MANAGER_LEADER_ELECTION=${ADDON_MANAGER_LEADER_ELECTION:-true}

# Remember that you can't log from functions that print some output (because
# logs are also printed on stdout).
# $1 level
# $2 message
function log() {
  # manage log levels manually here

  # add the timestamp if you find it useful
  case $1 in
    DB3 )
#        echo "$1: $2"
        ;;
    DB2 )
#        echo "$1: $2"
        ;;
    DBG )
#        echo "$1: $2"
        ;;
    INFO )
        echo "$1: $2"
        ;;
    WRN )
        echo "$1: $2"
        ;;
    ERR )
        echo "$1: $2"
        ;;
    * )
        echo "INVALID_LOG_LEVEL $1: $2"
        ;;
  esac
}

# Generate kubectl prune-allowlist flags from provided resource list.
function generate_prune_allowlist_flags() {
  local -r resources=( "$@" )
  for resource in "${resources[@]}"; do
    # Check if $resource isn't composed just of whitespaces by replacing ' '
    # with '' and checking whether the resulting string is not empty.
    if [[ -n "${resource// /}" ]]; then
      printf "%s" "--prune-allowlist ${resource} "
    fi
  done
}

# KUBECTL_EXTRA_PRUNE_WHITELIST is a list of extra allowed resources
# besides the default ones.
extra_prune_allowlist=
if [ -n "${KUBECTL_EXTRA_PRUNE_WHITELIST:-}" ]; then
  read -ra extra_prune_allowlist <<< "${KUBECTL_EXTRA_PRUNE_WHITELIST}"
fi
prune_allowlist=( "${KUBECTL_PRUNE_WHITELIST[@]}"  "${extra_prune_allowlist[@]}" )
prune_allowlist_flags=$(generate_prune_allowlist_flags "${prune_allowlist[@]}")

log INFO "== Generated kubectl prune allowlist flags: $prune_allowlist_flags =="

# $1 filename of addon to start.
# $2 count of tries to start the addon.
# $3 delay in seconds between two consecutive tries
# $4 namespace
function start_addon() {
  local -r addon_filename=$1;
  local -r tries=$2;
  local -r delay=$3;
  local -r namespace=$4

  create_resource_from_string "$(cat "${addon_filename}")" "${tries}" "${delay}" "${addon_filename}" "${namespace}"
}

# $1 string with json or yaml.
# $2 count of tries to start the addon.
# $3 delay in seconds between two consecutive tries
# $4 name of this object to use when logging about it.
# $5 namespace for this object
function create_resource_from_string() {
  local -r config_string=$1;
  local tries=$2;
  local -r delay=$3;
  local -r config_name=$4;
  local -r namespace=$5;
  while [ "${tries}" -gt 0 ]; do
    reconcile_resource_from_string "${config_string}" "${config_name}" "${namespace}" && \
      ensure_resource_from_string "${config_string}" "${config_name}" "${namespace}" && \
      return 0;
    (( tries-- ))
    log WRN "== Failed to start ${config_name} in namespace ${namespace} at $(date -Is). ${tries} tries remaining. =="
    sleep "${delay}";
  done
  return 1;
}

# Creates resources with addon mode Reconcile for create_resource_from_string.
# Does not perform pruning.
# $1 string with json or yaml.
# $2 name of this object for logging
# $3 namespace for the object
function reconcile_resource_from_string() {
  local -r config_string=$1;
  local -r config_name=$2;
  local -r namespace=$3;

  # kubectl_output must be declared ahead of time to allow capturing kubectl's exit code and not local's exit code.
  local kubectl_output;
  # shellcheck disable=SC2086
  # Disabling because "${KUBECTL_OPTS}" needs to allow for expansion here
  kubectl_output=$(echo "${config_string}" | ${KUBECTL} ${KUBECTL_OPTS} apply -f - \
    --namespace="${namespace}" -l ${ADDON_MANAGER_LABEL}=Reconcile 2>&1) && \
      log INFO "== Successfully reconciled ${config_name} in namespace ${namespace} at $(date -Is)" && \
      return 0;
  if echo "${kubectl_output}" | grep --silent "no objects"; then
    # Nothing to do.
    return 0;
  fi
  echo "${kubectl_output}" # for visibility of errors
  return 1;
}

# Creates resources with addon mode EnsureExists for create_resource_from_string.
# Does not perform pruning.
# $1 string with json or yaml.
# $2 name of this object for logging
# $3 namespace for the object
function ensure_resource_from_string() {
  local -r config_string=$1;
  local -r config_name=$2;
  local -r namespace=$3;

  # Resources that are set to the addon mode EnsureExists should not be overwritten if they already exist.
  local kubectl_output;
  # shellcheck disable=SC2086
  # Disabling because "${KUBECTL_OPTS}" needs to allow for expansion here
  kubectl_output=$(echo "${config_string}" | ${KUBECTL} ${KUBECTL_OPTS} create -f - \
    --namespace="${namespace}" -l ${ADDON_MANAGER_LABEL}=EnsureExists 2>&1) && \
      log INFO "== Successfully started ${config_name} in namespace ${namespace} at $(date -Is)" && \
      return 0;
  # Detect an already exists failure for creating EnsureExists resources.
  # All other errors should result in a retry.
  if echo "${kubectl_output}" | grep --silent "AlreadyExists"; then
    log INFO "== Skipping start ${config_name} in namespace ${namespace}, already exists at $(date -Is)"
    return 0;
  elif echo "${kubectl_output}" | grep --silent "no objects"; then
    # Nothing to do.
    return 0;
  fi
  echo "${kubectl_output}" # for visibility of errors
  return 1;
}

function reconcile_addons() {
  # TODO: Remove the first command in future release.
  # Adding this for backward compatibility. Old addons have CLUSTER_SERVICE_LABEL=true and don't have
  # ADDON_MANAGER_LABEL=EnsureExists will still be reconciled.
  # Filter out `configured` message to not noisily log.
  # `created`, `pruned` and errors will be logged.
  log INFO "== Reconciling with deprecated label =="
  # shellcheck disable=SC2086
  # Disabling because "${KUBECTL_OPTS}" needs to allow for expansion here
  ${KUBECTL} ${KUBECTL_OPTS} apply -f ${ADDON_PATH} \
    -l ${CLUSTER_SERVICE_LABEL}=true,${ADDON_MANAGER_LABEL}!=EnsureExists \
    --prune=true ${prune_allowlist_flags} --recursive | grep -v configured

  log INFO "== Reconciling with addon-manager label =="
  # shellcheck disable=SC2086
  # Disabling because "${KUBECTL_OPTS}" needs to allow for expansion here
  ${KUBECTL} ${KUBECTL_OPTS} apply -f ${ADDON_PATH} \
    -l ${CLUSTER_SERVICE_LABEL}!=true,${ADDON_MANAGER_LABEL}=Reconcile \
    --prune=true ${prune_allowlist_flags} --recursive | grep -v configured

  log INFO "== Kubernetes addon reconcile completed at $(date -Is) =="
}

function ensure_addons() {
  # Create objects already exist should fail.
  # Filter out `AlreadyExists` message to not noisily log.
  # shellcheck disable=SC2086
  # Disabling because "${KUBECTL_OPTS}" needs to allow for expansion here
  ${KUBECTL} ${KUBECTL_OPTS} create -f ${ADDON_PATH} \
    -l ${ADDON_MANAGER_LABEL}=EnsureExists --recursive 2>&1 | grep -v AlreadyExists

  log INFO "== Kubernetes addon ensure completed at $(date -Is) =="
}

function is_leader() {
  # In multi-master setup, only one addon manager should be running. We use
  # existing leader election in kube-controller-manager instead of implementing
  # a separate mechanism here.
  if ! $ADDON_MANAGER_LEADER_ELECTION; then
    log INFO "Leader election disabled."
    return 0;
  fi
  # shellcheck disable=SC2086
  # Disabling because "${KUBECTL_OPTS}" needs to allow for expansion here
  KUBE_CONTROLLER_MANAGER_LEADER=$(${KUBECTL} ${KUBECTL_OPTS} -n kube-system get leases.v1.coordination.k8s.io kube-controller-manager -o "jsonpath={.spec.holderIdentity}")

  case "${KUBE_CONTROLLER_MANAGER_LEADER}" in
  "")
    log ERR "No leader election info found."
    return 1
    ;;

  "${HOSTNAME}"_*)
    log INFO "Leader is $KUBE_CONTROLLER_MANAGER_LEADER"
    return 0
    ;;

  *)
    log INFO "Leader is $KUBE_CONTROLLER_MANAGER_LEADER, not ${HOSTNAME}_*"
    return 1
    ;;
  esac
}
