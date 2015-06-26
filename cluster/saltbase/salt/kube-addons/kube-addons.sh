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
KUBECTL=${KUBECTL_BIN:-/usr/local/bin/kubectl}

ADDON_CHECK_INTERVAL_SEC=${TEST_ADDON_CHECK_INTERVAL_SEC:-600}

SYSTEM_NAMESPACE=kube-system

function delete-if-present() {
  local -r rsrc=$1
  local -r name=$2
  local tries=$3
  local -r sleep=$4
  local -r namespace=$5

  while [ ${tries} -gt 0 ]; do
    # If we can't get the resource, either it doesn't exist, or its a transient failure.
    if ! ${KUBECTL} get ${rsrc} ${name}; then
       local count=$(${KUBECTL} get ${rsrc} ${name} 2> /dev/null | grep -c "not found")
       # It was not found, return
       if [ "${count}" != 0 ]; then
	 return
       fi
    fi
    if ${KUBECTL} stop ${rsrc} ${name}; then
      return
    fi
    echo "Failed to delete, waiting."
    sleep 5
    let tries=tries-1;
  done
}

function create-kubeconfig-secret() {
  local -r token=$1
  local -r username=$2
  local -r server=$3
  local -r safe_username=$(tr -s ':_' '--' <<< "${username}")

  # Make a kubeconfig file with the token.
  if [[ ! -z "${CA_CERT:-}" ]]; then
    # If the CA cert is available, put it into the secret rather than using
    # insecure-skip-tls-verify.
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
     server: ${server}
     certificate-authority-data: ${CA_CERT}
contexts:
- context:
    cluster: local
    user: ${username}
    namespace: ${SYSTEM_NAMESPACE} 
  name: service-account-context
current-context: service-account-context
EOF
  else
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
     server: ${server}
     insecure-skip-tls-verify: true
contexts:
- context:
    cluster: local
    user: ${username}
    namespace: ${SYSTEM_NAMESPACE}
  name: service-account-context
current-context: service-account-context
EOF
  fi

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
  create-resource-from-string "${secretyaml}" 100 10 "Secret-for-token-for-user-${username}" "${SYSTEM_NAMESPACE}" &
# TODO: label the secrets with special label so kubectl does not show these?
}

# $1 filename of addon to start.
# $2 count of tries to start the addon.
# $3 delay in seconds between two consecutive tries
# $4 namespace
function start_addon() {
  local -r addon_filename=$1;
  local -r tries=$2;
  local -r delay=$3;

  create-resource-from-string "$(cat ${addon_filename})" "${tries}" "${delay}" "${addon_filename}" $4
}

# $1 string with json or yaml.
# $2 count of tries to start the addon.
# $3 delay in seconds between two consecutive tries
# $4 name of this object to use when logging about it.
# $5 namespace for this object
function create-resource-from-string() {
  local -r config_string=$1;
  local tries=$2;
  local -r delay=$3;
  local -r config_name=$4;
  while [ ${tries} -gt 0 ]; do
    echo "${config_string}" | ${KUBECTL} --namespace=$5 create -f - && \
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
echo "== Kubernetes addon manager started at $(date -Is) with ADDON_CHECK_INTERVAL_SEC=${ADDON_CHECK_INTERVAL_SEC}=="

# Load the kube-env, which has all the environment variables we care
# about, in a flat yaml format.
kube_env_yaml="/var/cache/kubernetes-install/kube_env.yaml"
if [ ! -e "${kubelet_kubeconfig_file}" ]; then
  eval $(python -c '''
import pipes,sys,yaml

for k,v in yaml.load(sys.stdin).iteritems():
  print "readonly {var}={value}".format(var = k, value = pipes.quote(str(v)))
''' < "${kube_env_yaml}")
fi

# Delete old default namespaced resources.  Necessary for update
# TODO Needs testing
#for name in elasticsearch-logging-v1 kibana-logging-v1 kube-dns-v3 monitoring-heapster-v4 monitoring-influx-grafana-v1; do
#  delete-if-present rc ${name} 5 5 default
#done

#for name in elasticsearch-logging kibana-logging kube-dns monitoring-grafana monitoring-heapster monitoring-influxdb; do
#  delete-if-present service ${name} 5 5 default
#done

#for name in token-system-dns token-system-logging token-system-monitoring; do
#  delete-if-present secrets ${name} 5 5 default
#done

start_addon /etc/kubernetes/addons/namespace.yaml 100 10 default &

# Generate secrets for "internal service accounts".
# TODO(etune): move to a completely yaml/object based
# workflow so that service accounts can be created
# at the same time as the services that use them.
# NOTE: needs to run as root to read this file.
# Read each line in the csv file of tokens.
# Expect errors when the script is started again.
while read line; do
  # Split each line into the token and username.
  IFS=',' read -a parts <<< "${line}"
  token=${parts[0]}
  username=${parts[1]}
  # DNS is special, since it's necessary for cluster bootstrapping.
  if [[ "${username}" == "system:dns" ]] && [[ ! -z "${KUBERNETES_MASTER_NAME:-}" ]]; then
    create-kubeconfig-secret "${token}" "${username}" "https://${KUBERNETES_MASTER_NAME}"
  else
    # Set the server to https://kubernetes. Pods/components that
    # do not have DNS available will have to override the server.
    create-kubeconfig-secret "${token}" "${username}" "https://kubernetes.default.svc.cluster.local"
  fi
done < /srv/kubernetes/known_tokens.csv

# Create admission_control objects if defined before any other addon services. If the limits
# are defined in a namespace other than default, we should still create the limits for the
# default namespace.
for obj in $(find /etc/kubernetes/admission-controls \( -name \*.yaml -o -name \*.json \)); do
  start_addon ${obj} 100 10 default &
  echo "++ obj ${obj} is created ++"
done

# Check if the configuration has changed recently - in case the user
# created/updated/deleted the files on the master.
while true; do
  #kube-addon-update.sh must be deployed in the same directory as this file
  `dirname $0`/kube-addon-update.sh /etc/kubernetes/addons
  sleep $ADDON_CHECK_INTERVAL_SEC
done



