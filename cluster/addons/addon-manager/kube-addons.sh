#!/bin/bash

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

# The business logic for whether a given object should be created
# was already enforced by salt, and /etc/kubernetes/addons is the
# managed result is of that. Start everything below that directory.
KUBECTL=${KUBECTL_BIN:-/usr/local/bin/kubectl}
KUBECTL_OPTS=${KUBECTL_OPTS:-}

ADDON_CHECK_INTERVAL_SEC=${TEST_ADDON_CHECK_INTERVAL_SEC:-60}
ADDON_PATH=${ADDON_PATH:-/etc/kubernetes/addons}

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

# $1 filename of addon to start.
# $2 count of tries to start the addon.
# $3 delay in seconds between two consecutive tries
# $4 namespace
function start_addon() {
  local -r addon_filename=$1;
  local -r tries=$2;
  local -r delay=$3;
  local -r namespace=$4

  create_resource_from_string "$(cat ${addon_filename})" "${tries}" "${delay}" "${addon_filename}" "${namespace}"
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
  while [ ${tries} -gt 0 ]; do
    echo "${config_string}" | ${KUBECTL} ${KUBECTL_OPTS} --namespace="${namespace}" apply -f - && \
      log INFO "== Successfully started ${config_name} in namespace ${namespace} at $(date -Is)" && \
      return 0;
    let tries=tries-1;
    log WRN "== Failed to start ${config_name} in namespace ${namespace} at $(date -Is). ${tries} tries remaining. =="
    sleep ${delay};
  done
  return 1;
}

function reconcile_addons() {
  # TODO: Remove the first command in future release.
  # Adding this for backward compatibility. Old addons have CLUSTER_SERVICE_LABEL=true and don't have
  # ADDON_MANAGER_LABEL=EnsureExists will still be reconciled.
  # Filter out `configured` message to not noisily log.
  # `created`, `pruned` and errors will be logged.
  log INFO "== Reconciling with deprecated label =="
  ${KUBECTL} ${KUBECTL_OPTS} apply --namespace=${SYSTEM_NAMESPACE} -f ${ADDON_PATH} \
    -l ${CLUSTER_SERVICE_LABEL}=true,${ADDON_MANAGER_LABEL}!=EnsureExists \
    --prune=true --recursive | grep -v configured

  log INFO "== Reconciling with addon-manager label =="
  ${KUBECTL} ${KUBECTL_OPTS} apply --namespace=${SYSTEM_NAMESPACE} -f ${ADDON_PATH} \
    -l ${CLUSTER_SERVICE_LABEL}!=true,${ADDON_MANAGER_LABEL}=Reconcile \
    --prune=true --recursive | grep -v configured

  log INFO "== Kubernetes addon reconcile completed at $(date -Is) =="
}

function ensure_addons() {
  # Create objects already exist should fail.
  # Filter out `AlreadyExists` message to not noisily log.
  ${KUBECTL} ${KUBECTL_OPTS} create --namespace=${SYSTEM_NAMESPACE} -f ${ADDON_PATH} \
    -l ${ADDON_MANAGER_LABEL}=EnsureExists --recursive 2>&1 | grep -v AlreadyExists

  log INFO "== Kubernetes addon ensure completed at $(date -Is) =="
}

# The business logic for whether a given object should be created
# was already enforced by salt, and /etc/kubernetes/addons is the
# managed result is of that. Start everything below that directory.
log INFO "== Kubernetes addon manager started at $(date -Is) with ADDON_CHECK_INTERVAL_SEC=${ADDON_CHECK_INTERVAL_SEC} =="

# Create the namespace that will be used to host the cluster-level add-ons.
start_addon /opt/namespace.yaml 100 10 "" &

# Wait for the default service account to be created in the kube-system namespace.
token_found=""
while [ -z "${token_found}" ]; do
  sleep .5
  token_found=$(${KUBECTL} ${KUBECTL_OPTS} get --namespace="${SYSTEM_NAMESPACE}" serviceaccount default -o go-template="{{with index .secrets 0}}{{.name}}{{end}}")
  if [[ $? -ne 0 ]]; then
    token_found="";
    log WRN "== Error getting default service account, retry in 0.5 second =="
  fi
done

log INFO "== Default service account in the ${SYSTEM_NAMESPACE} namespace has token ${token_found} =="

# Create admission_control objects if defined before any other addon services. If the limits
# are defined in a namespace other than default, we should still create the limits for the
# default namespace.
for obj in $(find /etc/kubernetes/admission-controls \( -name \*.yaml -o -name \*.json \)); do
  start_addon "${obj}" 100 10 default &
  log INFO "++ obj ${obj} is created ++"
done

# Start the apply loop.
# Check if the configuration has changed recently - in case the user
# created/updated/deleted the files on the master.
log INFO "== Entering periodical apply loop at $(date -Is) =="
while true; do
  start_sec=$(date +"%s")
  ensure_addons
  reconcile_addons
  end_sec=$(date +"%s")
  len_sec=$((${end_sec}-${start_sec}))
  # subtract the time passed from the sleep time
  if [[ ${len_sec} -lt ${ADDON_CHECK_INTERVAL_SEC} ]]; then
    sleep_time=$((${ADDON_CHECK_INTERVAL_SEC}-${len_sec}))
    sleep ${sleep_time}
  fi
done
