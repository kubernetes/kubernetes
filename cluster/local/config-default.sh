#!/bin/bash

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

export DOCKER_OPTS=${DOCKER_OPTS:-""}
export DOCKER_NATIVE=${DOCKER_NATIVE:-""}
export DOCKER=(docker ${DOCKER_OPTS})
export DOCKERIZE_KUBELET=${DOCKERIZE_KUBELET:-""}
export ALLOW_PRIVILEGED=${ALLOW_PRIVILEGED:-""}
export ALLOW_SECURITY_CONTEXT=${ALLOW_SECURITY_CONTEXT:-""}
export RUNTIME_CONFIG=${RUNTIME_CONFIG:-""}
export NET_PLUGIN=${NET_PLUGIN:-""}
export NET_PLUGIN_DIR=${NET_PLUGIN_DIR:-""}
export KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
# We disable cluster DNS by default because this script uses docker0 (or whatever
# container bridge docker is currently using) and we don't know the IP of the
# DNS pod to pass in as --cluster-dns. To set this up by hand, set this flag
# and change DNS_SERVER_IP to the appropriate IP.
export ENABLE_CLUSTER_DNS=${KUBE_ENABLE_CLUSTER_DNS:-false}
export DNS_SERVER_IP=${KUBE_DNS_SERVER_IP:-10.0.0.10}
export DNS_DOMAIN=${KUBE_DNS_NAME:-"cluster.local"}
export DNS_REPLICAS=${KUBE_DNS_REPLICAS:-1}
export KUBECTL=${KUBECTL:-cluster/kubectl.sh}
export WAIT_FOR_URL_API_SERVER=${WAIT_FOR_URL_API_SERVER:-10}
# ENABLE_DAEMON=${ENABLE_DAEMON:-false}
export ENABLE_DAEMON=${ENABLE_DAEMON:-true}
export HOSTNAME_OVERRIDE=${HOSTNAME_OVERRIDE:-"127.0.0.1"}
export CLOUD_PROVIDER=${CLOUD_PROVIDER:-""}

export API_PORT=${API_PORT:-8080}
export API_HOST=${API_HOST:-127.0.0.1}
export KUBELET_HOST=${KUBELET_HOST:-"127.0.0.1"}
# By default only allow CORS for requests on localhost
export API_CORS_ALLOWED_ORIGINS=${API_CORS_ALLOWED_ORIGINS:-"/127.0.0.1(:[0-9]+)?$,/localhost(:[0-9]+)?$"}
export KUBELET_PORT=${KUBELET_PORT:-10250}
export LOG_LEVEL=${LOG_LEVEL:-3}
export CONTAINER_RUNTIME=${CONTAINER_RUNTIME:-"docker"}
export RKT_PATH=${RKT_PATH:-""}
export RKT_STAGE1_IMAGE=${RKT_STAGE1_IMAGE:-""}
export CHAOS_CHANCE=${CHAOS_CHANCE:-0.0}
export CPU_CFS_QUOTA=${CPU_CFS_QUOTA:-false}
export ENABLE_HOSTPATH_PROVISIONER=${ENABLE_HOSTPATH_PROVISIONER:-"false"}
export CLAIM_BINDER_SYNC_PERIOD=${CLAIM_BINDER_SYNC_PERIOD:-"10m"} # current k8s default

export APISERVER_PIDFILE=${APISERVER_PIDFILE:-"/tmp/kube-apiserver.pid"}
export CTLRMGR_PIDFILE=${CTLRMGR_PIDFILE:-"/tmp/kube-controller-manager.pid"}
export KUBELET_PIDFILE=${KUBELET_PIDFILE:-"/tmp/kubelet.pid"}
export PROXY_PIDFILE=${PROXY_PIDFILE:-"/tmp/kube-proxy.pid"}
export SCHEDULER_PIDFILE=${SCHEDULER_PIDFILE:-"/tmp/kube-scheduler.pid"}

export GO_OUT=""

# It's impossible to create a multinode cluster by local provider
export NUM_NODES=1
