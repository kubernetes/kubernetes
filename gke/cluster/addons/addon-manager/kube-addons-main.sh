#!/usr/bin/env bash

# Copyright 2020 The Kubernetes Authors.
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

# Import required functions. The addon manager is installed to /opt in
# production use (see the Dockerfile)
# Disabling shellcheck following files as the full path would be required.
if [ -f "kube-addons.sh" ]; then
  # shellcheck disable=SC1091
  source "kube-addons.sh"
elif [ -f "/opt/kube-addons.sh" ]; then
  # shellcheck disable=SC1091
  source "/opt/kube-addons.sh"
else
  # If the required source is missing, we have to fail.
  log ERR "== Could not find kube-addons.sh (not in working directory or /opt) at $(date -Is) =="
  exit 1
fi

# The business logic for whether a given object should be created
# was already enforced by salt, and /etc/kubernetes/addons is the
# managed result of that. Start everything below that directory.
log INFO "== Kubernetes addon manager started at $(date -Is) with ADDON_CHECK_INTERVAL_SEC=${ADDON_CHECK_INTERVAL_SEC} =="

# Wait for the default service account to be created in the kube-system namespace.
token_found=""
while [ -z "${token_found}" ]; do
  sleep .5
  # shellcheck disable=SC2086
  # Disabling because "${KUBECTL_OPTS}" needs to allow for expansion here
  if ! token_found=$(${KUBECTL} ${KUBECTL_OPTS} get --namespace="${SYSTEM_NAMESPACE}" serviceaccount default -o go-template="{{with index .secrets 0}}{{.name}}{{end}}"); then
    token_found="";
    log WRN "== Error getting default service account, retry in 0.5 second =="
  fi
done

log INFO "== Default service account in the ${SYSTEM_NAMESPACE} namespace has token ${token_found} =="

# Create admission_control objects if defined before any other addon services. If the limits
# are defined in a namespace other than default, we should still create the limits for the
# default namespace.
while IFS=$'\n' read -r obj; do
  start_addon "${obj}" 100 10 default &
  log INFO "++ obj ${obj} is created ++"
done < <(find /etc/kubernetes/admission-controls \( -name \*.yaml -o -name \*.json \))

# Start the apply loop.
# Check if the configuration has changed recently - in case the user
# created/updated/deleted the files on the master.
log INFO "== Entering periodical apply loop at $(date -Is) =="
while true; do
  start_sec=$(date +"%s")
  if is_leader; then
    ensure_addons
    reconcile_addons
  else
    log INFO "Not elected leader, going back to sleep."
  fi
  end_sec=$(date +"%s")
  len_sec=$((end_sec-start_sec))
  # subtract the time passed from the sleep time
  if [[ ${len_sec} -lt ${ADDON_CHECK_INTERVAL_SEC} ]]; then
    sleep_time=$((ADDON_CHECK_INTERVAL_SEC-len_sec))
    sleep ${sleep_time}
  fi
done
