#!/bin/bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
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

# The business logic for whether a given object should be created
# was already enforced by salt, and /etc/kubernetes/addons is the
# managed result is of that. Start everything below that directory.
KUBECTL=kubectl

function create-kubeconfig-secret() {
  local -r token=$1
  local -r username=$2
  local -r safe_username=$(tr -s ':_' '--' <<< "${username}")

  # Make a kubeconfig file with the token.
  # TODO(etune): put apiserver certs into secret too, and reference from authfile,
  # so that "Insecure" is not needed.
  # Point the kubeconfig file at https://kubernetes:443. Pods/components that
  # do not have DNS available will have to override the server.
  read -r -d '' kubeconfig <<EOF
apiVersion: v1
kind: Config
users:
- name: ${username}
  user:
    token: ${token}
clusters:
- name: local
  cluster:
     server: "https://kubernetes:443"
     insecure-skip-tls-verify: true
contexts:
- context:
    cluster: local
    user: ${username}
  name: service-account-context
current-context: service-account-context
EOF
  local -r kubeconfig_base64=$(echo "${kubeconfig}" | base64 -w0)
  read -r -d '' secretyaml <<EOF
apiVersion: v1beta3
data:
  kubeconfig: ${kubeconfig_base64}
kind: Secret
metadata:
  name: token-${safe_username}
type: Opaque
EOF
  create-resource-from-string "${secretyaml}" 100 10 "Secret-for-token-for-user-${username}" &
# TODO: label the secrets with special label so kubectl does not show these?
}

# $1 filename of addon to start.
# $2 count of tries to start the addon.
# $3 delay in seconds between two consecutive tries
function start_addon() {
  local -r addon_filename=$1;
  local -r tries=$2;
  local -r delay=$3;

  create-resource-from-string "$(cat ${addon_filename})" "${tries}" "${delay}" "${addon_filename}"
}

# $1 string with json or yaml.
# $2 count of tries to start the addon.
# $3 delay in seconds between two consecutive tries
# $3 name of this object to use when logging about it.
function create-resource-from-string() {
  local -r config_string=$1;
  local tries=$2;
  local -r delay=$3;
  local -r config_name=$1;
  while [ ${tries} -gt 0 ]; do
    echo "${config_string}" | ${KUBECTL} create -f - && \
        echo "== Successfully started ${config_name} at $(date -Is)" && \
        return 0;
    let tries=tries-1;
    echo "== Failed to start ${config_name} at $(date -Is). ${tries} tries remaining. =="
    sleep ${delay};
  done
  return 1;
}

# The business logic for whether a given object should be created
# was already enforced by salt, and /etc/kubernetes/addons is the
# managed result is of that. Start everything below that directory.
echo "== Kubernetes addon manager started at $(date -Is) =="
touch /srv/kubernetes/known_tokens.csv
echo -n > /srv/kubernetes/known_tokens.csv
# Generate tokens for other "service accounts".  Append to known_tokens.
#
# NB: If this list ever changes, this script actually has to
# change to detect the existence of this file, kill any deleted
# old tokens and add any new tokens (to handle the upgrade case).
service_accounts=("system:scheduler" "system:controller_manager" "system:logging" "system:monitoring" "system:dns")
for account in "${service_accounts[@]}"; do
  token=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
  echo "${token},${account},${account}" >> /srv/kubernetes/known_tokens.csv
done

# Generate secrets for "internal service accounts".
# TODO(etune): move to a completely yaml/object based
# workflow so that service accounts can be created
# at the same time as the services that use them.
# NOTE: needs to run as root to read this file.
# Read each line in the csv file of tokens.
while read line; do
  # Split each line into the token and username.
  IFS=',' read -a parts <<< "${line}"
  token=${parts[0]}
  username=${parts[1]}
  create-kubeconfig-secret "${token}" "${username}"
done < /srv/kubernetes/known_tokens.csv

# Create admission_control objects if defined before any other addon services. If the limits
# are defined in a namespace other than default, we should still create the limits for the
# default namespace.
for obj in $(find /etc/kubernetes/admission-controls \( -name \*.yaml -o -name \*.json \)); do
  start_addon ${obj} 100 10 &
  echo "++ obj ${obj} is created ++"
done

for obj in $(find /etc/kubernetes/addons \( -name \*.yaml -o -name \*.json \)); do
  start_addon ${obj} 100 10 &
  echo "++ addon ${obj} starting in pid $! ++"
done
noerrors="true"
for pid in $(jobs -p); do
  wait ${pid} || noerrors="false"
  echo "++ pid ${pid} complete ++"
done
if [ ${noerrors} == "true" ]; then
  echo "== Kubernetes addon manager completed successfully at $(date -Is) =="
else
  echo "== Kubernetes addon manager completed with errors at $(date -Is) =="
fi

# We stay around so that status checks by salt make it look like
# the service is good. (We could do this is other ways, but this
# is simple.)
sleep infinity