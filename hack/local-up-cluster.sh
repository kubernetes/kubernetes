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

KUBE_VERBOSE=${KUBE_VERBOSE:-1}
if (( KUBE_VERBOSE > 4 )); then
  set -x
fi

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
cd "${KUBE_ROOT}"

# This script builds and runs a local kubernetes cluster. You may need to run
# this as root to allow kubelet to open containerd's socket, and to write the test
# CA in /var/run/kubernetes.
# Usage: `hack/local-up-cluster.sh`.

# Dump config at KUBE_VERBOSE >= 2.
if (( KUBE_VERBOSE >= 2 )); then
  set -x
fi

ALLOW_PRIVILEGED=${ALLOW_PRIVILEGED:-""}
RUNTIME_CONFIG=${RUNTIME_CONFIG:-""}
KUBELET_AUTHORIZATION_WEBHOOK=${KUBELET_AUTHORIZATION_WEBHOOK:-""}
KUBELET_AUTHENTICATION_WEBHOOK=${KUBELET_AUTHENTICATION_WEBHOOK:-""}
POD_MANIFEST_PATH=${POD_MANIFEST_PATH:-"/var/run/kubernetes/static-pods"}
KUBELET_FLAGS=${KUBELET_FLAGS:-""}
KUBELET_IMAGE=${KUBELET_IMAGE:-""}
# many dev environments run with swap on, so we don't fail in this env
FAIL_SWAP_ON=${FAIL_SWAP_ON:-"false"}
# Name of the dns addon, eg: "kube-dns" or "coredns"
DNS_ADDON=${DNS_ADDON:-"coredns"}
CLUSTER_CIDR=${CLUSTER_CIDR:-10.1.0.0/16}
SERVICE_CLUSTER_IP_RANGE=${SERVICE_CLUSTER_IP_RANGE:-10.0.0.0/24}
FIRST_SERVICE_CLUSTER_IP=${FIRST_SERVICE_CLUSTER_IP:-10.0.0.1}
# if enabled, must set CGROUP_ROOT
CGROUPS_PER_QOS=${CGROUPS_PER_QOS:-true}
# name of the cgroup driver, i.e. cgroupfs or systemd
CGROUP_DRIVER=${CGROUP_DRIVER:-""}
# if cgroups per qos is enabled, optionally change cgroup root
CGROUP_ROOT=${CGROUP_ROOT:-""}
# owner of client certs, default to current user if not specified
USER=${USER:-$(whoami)}
# if true, limited swap is being used instead of no swap (default)
LIMITED_SWAP=${LIMITED_SWAP:-""}

# required for cni installation
CNI_CONFIG_DIR=${CNI_CONFIG_DIR:-/etc/cni/net.d}
CNI_PLUGINS_VERSION=${CNI_PLUGINS_VERSION:-"v1.8.0"}
# The arch of the CNI binary, if not set, will be fetched based on the value of `uname -m`
CNI_TARGETARCH=${CNI_TARGETARCH:-""}
CNI_PLUGINS_URL="https://github.com/containernetworking/plugins/releases/download"

# enables testing eviction scenarios locally.
EVICTION_HARD=${EVICTION_HARD:-"imagefs.available<15%,memory.available<100Mi,nodefs.available<10%,nodefs.inodesFree<5%"}
EVICTION_SOFT=${EVICTION_SOFT:-""}
EVICTION_PRESSURE_TRANSITION_PERIOD=${EVICTION_PRESSURE_TRANSITION_PERIOD:-"1m"}

# This script uses docker0 (or whatever container bridge docker is currently using)
# and we don't know the IP of the DNS pod to pass in as --cluster-dns.
# To set this up by hand, set this flag and change DNS_SERVER_IP.
# Note also that you need API_HOST (defined below) for correct DNS.
KUBE_PROXY_MODE=${KUBE_PROXY_MODE:-""}
ENABLE_CLUSTER_DNS=${KUBE_ENABLE_CLUSTER_DNS:-true}
ENABLE_NODELOCAL_DNS=${KUBE_ENABLE_NODELOCAL_DNS:-false}
DNS_SERVER_IP=${KUBE_DNS_SERVER_IP:-10.0.0.10}
LOCAL_DNS_IP=${KUBE_LOCAL_DNS_IP:-169.254.20.10}
DNS_MEMORY_LIMIT=${KUBE_DNS_MEMORY_LIMIT:-170Mi}
DNS_DOMAIN=${KUBE_DNS_NAME:-"cluster.local"}
WAIT_FOR_URL_API_SERVER=${WAIT_FOR_URL_API_SERVER:-60}
MAX_TIME_FOR_URL_API_SERVER=${MAX_TIME_FOR_URL_API_SERVER:-1}
ENABLE_DAEMON=${ENABLE_DAEMON:-false}
HOSTNAME_OVERRIDE=${HOSTNAME_OVERRIDE:-"127.0.0.1"}
EXTERNAL_CLOUD_PROVIDER=${EXTERNAL_CLOUD_PROVIDER:-false}
EXTERNAL_CLOUD_PROVIDER_BINARY=${EXTERNAL_CLOUD_PROVIDER_BINARY:-""}
EXTERNAL_CLOUD_VOLUME_PLUGIN=${EXTERNAL_CLOUD_VOLUME_PLUGIN:-""}
CONFIGURE_CLOUD_ROUTES=${CONFIGURE_CLOUD_ROUTES:-true}
CLOUD_CTLRMGR_FLAGS=${CLOUD_CTLRMGR_FLAGS:-""}
CLOUD_PROVIDER=${CLOUD_PROVIDER:-""}
CLOUD_CONFIG=${CLOUD_CONFIG:-""}
KUBELET_PROVIDER_ID=${KUBELET_PROVIDER_ID:-"$(hostname)"}
FEATURE_GATES=${FEATURE_GATES:-"AllAlpha=false"}
EMULATED_VERSION=${EMULATED_VERSION:+kube=$EMULATED_VERSION}
TOPOLOGY_MANAGER_POLICY=${TOPOLOGY_MANAGER_POLICY:-""}
CPUMANAGER_POLICY=${CPUMANAGER_POLICY:-""}
CPUMANAGER_RECONCILE_PERIOD=${CPUMANAGER_RECONCILE_PERIOD:-""}
CPUMANAGER_POLICY_OPTIONS=${CPUMANAGER_POLICY_OPTIONS:-""}
LEADER_ELECT=${LEADER_ELECT:-false}
STORAGE_BACKEND=${STORAGE_BACKEND:-"etcd3"}
STORAGE_MEDIA_TYPE=${STORAGE_MEDIA_TYPE:-"application/vnd.kubernetes.protobuf"}
# preserve etcd data. you also need to set ETCD_DIR.
PRESERVE_ETCD="${PRESERVE_ETCD:-false}"
ENABLE_TRACING=${ENABLE_TRACING:-false}

# enable Kubernetes-CSI snapshotter
ENABLE_CSI_SNAPSHOTTER=${ENABLE_CSI_SNAPSHOTTER:-false}

KUBECONFIG_TOKEN=${KUBECONFIG_TOKEN:-""}
AUTH_ARGS=${AUTH_ARGS:-""}

# WebHook Authentication and Authorization
AUTHORIZATION_WEBHOOK_CONFIG_FILE=${AUTHORIZATION_WEBHOOK_CONFIG_FILE:-""}
AUTHENTICATION_WEBHOOK_CONFIG_FILE=${AUTHENTICATION_WEBHOOK_CONFIG_FILE:-""}

# Install a default storage class (enabled by default)
DEFAULT_STORAGE_CLASS=${KUBE_DEFAULT_STORAGE_CLASS:-true}

# Do not run the mutation detector by default on a local cluster.
# It is intended for a specific type of testing and inherently leaks memory.
KUBE_CACHE_MUTATION_DETECTOR="${KUBE_CACHE_MUTATION_DETECTOR:-false}"
export KUBE_CACHE_MUTATION_DETECTOR

# panic the server on watch decode errors since they are considered coder mistakes
KUBE_PANIC_WATCH_DECODE_ERROR="${KUBE_PANIC_WATCH_DECODE_ERROR:-true}"
export KUBE_PANIC_WATCH_DECODE_ERROR

# Default list of admission Controllers to invoke prior to persisting objects in cluster
# The order defined here does not matter.
ENABLE_ADMISSION_PLUGINS=${ENABLE_ADMISSION_PLUGINS:-"NamespaceLifecycle,LimitRanger,ServiceAccount,DefaultStorageClass,DefaultTolerationSeconds,Priority,MutatingAdmissionWebhook,ValidatingAdmissionWebhook,ResourceQuota,NodeRestriction"}
DISABLE_ADMISSION_PLUGINS=${DISABLE_ADMISSION_PLUGINS:-""}
ADMISSION_CONTROL_CONFIG_FILE=${ADMISSION_CONTROL_CONFIG_FILE:-""}

# START_MODE can be 'all', 'kubeletonly', 'nokubelet', 'nokubeproxy', or 'nokubelet,nokubeproxy'
if [[ -z "${START_MODE:-}" ]]; then
    case "$(uname -s)" in
      Darwin)
        START_MODE=nokubelet,nokubeproxy
        ;;
      Linux)
        START_MODE=all
        ;;
      *)
        echo "Unsupported host OS.  Must be Linux or Mac OS X." >&2
        exit 1
        ;;
    esac
fi

# A list of controllers to enable
KUBE_CONTROLLERS="${KUBE_CONTROLLERS:-"*"}"

# Audit policy
AUDIT_POLICY_FILE=${AUDIT_POLICY_FILE:-""}

# dmesg command PID for cleanup
DMESG_PID=${DMESG_PID:-""}

# Stop logging commands again at KUBE_VERBOSE <= 4.
if (( KUBE_VERBOSE <= 4 )); then
  set +x
fi

# Stop right away if the build fails
set -e

source "${KUBE_ROOT}/hack/lib/init.sh"
kube::util::ensure-gnu-sed

function usage {
            echo "This script starts a local kube cluster. "
            echo "Example 0: hack/local-up-cluster.sh -h  (this 'help' usage description)"
            echo "Example 1: hack/local-up-cluster.sh -o _output/dockerized/bin/linux/amd64/ (run from docker output)"
            echo "Example 2: hack/local-up-cluster.sh -O (auto-guess the bin path for your platform)"
            echo "Example 3: hack/local-up-cluster.sh (build a local copy of the source)"
            echo "Example 4: TOPOLOGY_MANAGER_POLICY=\"single-numa-node\" \\"
            echo "           CPUMANAGER_POLICY=\"static\" \\"
            echo "           CPUMANAGER_POLICY_OPTIONS=full-pcpus-only=\"true\" \\"
            echo "           CPUMANAGER_RECONCILE_PERIOD=\"5s\" \\"
            echo "           KUBELET_FLAGS=\"--kube-reserved=cpu=1,memory=2Gi,ephemeral-storage=1Gi --system-reserved=cpu=1,memory=2Gi,ephemeral-storage=1Gi\" \\"
            echo "           hack/local-up-cluster.sh (build a local copy of the source with full-pcpus-only CPU Management policy)"
            echo ""
            echo "-d         dry-run: prepare for running commands, then show their command lines instead of running them"
}

# This function guesses where the existing cached binary build is for the `-O`
# flag
function guess_built_binary_path {
  local apiserver_path
  apiserver_path=$(kube::util::find-binary "kube-apiserver")
  if [[ -z "${apiserver_path}" ]]; then
    return
  fi
  echo -n "$(dirname "${apiserver_path}")"
}

### Allow user to supply the source directory.
GO_OUT=${GO_OUT:-}
DRY_RUN=
while getopts "dho:O" OPTION
do
    case ${OPTION} in
        d)
            echo "skipping running commands"
            DRY_RUN=1
            ;;
        o)
            echo "skipping build"
            GO_OUT="${OPTARG}"
            echo "using source ${GO_OUT}"
            ;;
        O)
            GO_OUT=$(guess_built_binary_path)
            if [ "${GO_OUT}" == "" ]; then
                echo "Could not guess the correct output directory to use."
                exit 1
            fi
            ;;
        h)
            usage
            exit
            ;;
        ?)
            usage
            exit
            ;;
    esac
done

# run executes the command specified by its parameters if DRY_RUN is empty,
# otherwise it prints them.
#
# The first parameter must be the name of the Kubernetes components.
# It is only used when printing the command in dry-run mode.
# The second parameter is a log file for the command. It may be empty.
function run {
    local what="$1"
    local log="$2"
    shift
    shift
    if [[ -z "${DRY_RUN}" ]]; then
        if [[ -z "${log}" ]]; then
            "${@}"
        else
            "${@}" >"${log}" 2>&1
        fi
    else
        echo "RUN ${what}: ${*}"
    fi
}

if [ -z "${GO_OUT}" ]; then
    binaries_to_build="cmd/kubectl cmd/kube-apiserver cmd/kube-controller-manager cmd/cloud-controller-manager cmd/kube-scheduler"
    if [[ "${START_MODE}" != *"nokubelet"* ]]; then
      binaries_to_build="${binaries_to_build} cmd/kubelet"
    fi
    if [[ "${START_MODE}" != *"nokubeproxy"* ]]; then
      binaries_to_build="${binaries_to_build} cmd/kube-proxy"
    fi
    make -C "${KUBE_ROOT}" WHAT="${binaries_to_build}"
else
    echo "skipped the build because GO_OUT was set (${GO_OUT})"
fi

# Shut down anyway if there's an error.
set +e

if (( KUBE_VERBOSE >= 2 )); then
  set -x
fi

# Ports opened by the different components.
# They must be available on the API_BIND_ADDR (same for all components for the sake of simplicity).
# Sorted by default port number!
API_PORT=${API_PORT:-0}
API_SECURE_PORT=${API_SECURE_PORT:-6443}
KUBELET_HEALTHZ_PORT=${KUBELET_HEALTHZ_PORT:-10248}
PROXY_METRICS_PORT=${PROXY_METRICS_PORT:-10249}
KUBELET_PORT=${KUBELET_PORT:-10250}
# By default we use 0(close it) for it's insecure
KUBELET_READ_ONLY_PORT=${KUBELET_READ_ONLY_PORT:-0}
PROXY_HEALTHZ_PORT=${PROXY_HEALTHZ_PORT:-10256}
KCM_SECURE_PORT=${KCM_SECURE_PORT:-10257}
SCHEDULER_SECURE_PORT=${SCHEDULER_SECURE_PORT:-10259}

# WARNING: For DNS to work on most setups you should export API_HOST as the docker0 ip address,
API_HOST=${API_HOST:-localhost}
API_HOST_IP=${API_HOST_IP:-"127.0.0.1"} # Used for all components, except kubelet.
KUBELET_HOST=${KUBELET_HOST:-"127.0.0.1"} # Also the bind address.
ADVERTISE_ADDRESS=${ADVERTISE_ADDRESS:-""}
NODE_PORT_RANGE=${NODE_PORT_RANGE:-""}
API_BIND_ADDR=${API_BIND_ADDR:-"0.0.0.0"} # Used for all components, except kubelet.
EXTERNAL_HOSTNAME=${EXTERNAL_HOSTNAME:-localhost}

KUBELET_RESOLV_CONF=${KUBELET_RESOLV_CONF:-"/etc/resolv.conf"}
# By default only allow CORS for requests on localhost
API_CORS_ALLOWED_ORIGINS=${API_CORS_ALLOWED_ORIGINS:-//127.0.0.1(:[0-9]+)?$,//localhost(:[0-9]+)?$}
LOG_LEVEL=${LOG_LEVEL:-3}
# Use to increase verbosity on particular files, e.g. LOG_SPEC=token_controller*=5,other_controller*=4
LOG_SPEC=${LOG_SPEC:-""}
LOG_DIR=${LOG_DIR:-"/tmp"}
TMP_DIR=${TMP_DIR:-$(kube::realpath "$(mktemp -d -t "$(basename "$0").XXXXXX")")}
CONTAINER_RUNTIME_ENDPOINT=${CONTAINER_RUNTIME_ENDPOINT:-"unix:///run/containerd/containerd.sock"}
RUNTIME_REQUEST_TIMEOUT=${RUNTIME_REQUEST_TIMEOUT:-"2m"}
IMAGE_SERVICE_ENDPOINT=${IMAGE_SERVICE_ENDPOINT:-""}
CPU_CFS_QUOTA=${CPU_CFS_QUOTA:-true}
ENABLE_HOSTPATH_PROVISIONER=${ENABLE_HOSTPATH_PROVISIONER:-"false"}
CLAIM_BINDER_SYNC_PERIOD=${CLAIM_BINDER_SYNC_PERIOD:-"15s"} # current k8s default
ENABLE_CONTROLLER_ATTACH_DETACH=${ENABLE_CONTROLLER_ATTACH_DETACH:-"true"} # current default
LOCAL_STORAGE_CAPACITY_ISOLATION=${LOCAL_STORAGE_CAPACITY_ISOLATION:-"true"} # current default
# This is the default dir and filename where the apiserver will generate a self-signed cert
# which should be able to be used as the CA to verify itself
CERT_DIR=${CERT_DIR:-"/var/run/kubernetes"}
ROOT_CA_FILE=${CERT_DIR}/server-ca.crt
CLUSTER_SIGNING_CERT_FILE=${CLUSTER_SIGNING_CERT_FILE:-"${CERT_DIR}/client-ca.crt"}
CLUSTER_SIGNING_KEY_FILE=${CLUSTER_SIGNING_KEY_FILE:-"${CERT_DIR}/client-ca.key"}
# Reuse certs will skip generate new ca/cert files under CERT_DIR
# it's useful with PRESERVE_ETCD=true because new ca will make existed service account secrets invalided
REUSE_CERTS=${REUSE_CERTS:-false}


# Ensure CERT_DIR is created for auto-generated crt/key and kubeconfig
mkdir -p "${CERT_DIR}" &>/dev/null || sudo mkdir -p "${CERT_DIR}"
CONTROLPLANE_SUDO=$(test -w "${CERT_DIR}" || echo "sudo -E")

if (( KUBE_VERBOSE <= 4 )); then
  set +x
fi

function test_apiserver_off {
    # For the common local scenario, fail fast if server is already running.
    # this can happen if you run local-up-cluster.sh twice and kill etcd in between.
    if [[ "${API_PORT}" -gt "0" ]]; then
        if ! curl --silent -g "${API_HOST}:${API_PORT}" ; then
            echo "API SERVER insecure port is free, proceeding..."
        else
            echo "ERROR starting API SERVER, exiting. Some process on ${API_HOST} is serving already on ${API_PORT}"
            exit 1
        fi
    fi

    if ! curl --silent -k -g "${API_HOST}:${API_SECURE_PORT}" ; then
        echo "API SERVER secure port is free, proceeding..."
    else
        echo "ERROR starting API SERVER, exiting. Some process on ${API_HOST} is serving already on ${API_SECURE_PORT}"
        exit 1
    fi
}

function detect_arch {
    local host_arch

    case "$(uname -m)" in
      x86_64*)
        host_arch=amd64
        ;;
      i?86_64*)
        host_arch=amd64
        ;;
      amd64*)
        host_arch=amd64
        ;;
      aarch64*)
        host_arch=arm64
        ;;
      arm64*)
        host_arch=arm64
        ;;
      arm*)
        host_arch=arm
        ;;
      i?86*)
        host_arch=x86
        ;;
      s390x*)
        host_arch=s390x
        ;;
      ppc64le*)
        host_arch=ppc64le
        ;;
      *)
        echo "Unsupported host arch. Must be x86_64, 386, arm, arm64, s390x or ppc64le." >&2
        exit 1
        ;;
    esac

  if [[ -z "${host_arch}" ]]; then
    return
  fi
  echo -n "${host_arch}"
}

function detect_os {
    local host_os

    case "$(uname -s)" in
      Darwin)
        host_os=darwin
        ;;
      Linux)
        host_os=linux
        ;;
      *)
        echo "Unsupported host OS.  Must be Linux or Mac OS X." >&2
        exit 1
        ;;
    esac

  if [[ -z "${host_os}" ]]; then
    return
  fi
  echo -n "${host_os}"
}

function detect_binary {
    host_arch=$(detect_arch)
    host_os=$(detect_os)

    GO_OUT="${KUBE_ROOT}/_output/local/bin/${host_os}/${host_arch}"
}

cleanup()
{
  echo "Cleaning up..."
  # delete running images
  # if [[ "${ENABLE_CLUSTER_DNS}" == true ]]; then
  # Still need to figure why this commands throw an error: Error from server: client: etcd cluster is unavailable or misconfigured
  #     ${KUBECTL} --namespace=kube-system delete service kube-dns
  # And this one hang forever:
  #     ${KUBECTL} --namespace=kube-system delete rc kube-dns-v10
  # fi

  # Check if the API server is still running
  [[ -n "${APISERVER_PID-}" ]] && kube::util::read-array APISERVER_PIDS < <(pgrep -P "${APISERVER_PID}" ; ps -o pid= -p "${APISERVER_PID}")
  [[ -n "${APISERVER_PIDS-}" ]] && sudo kill "${APISERVER_PIDS[@]}" 2>/dev/null

  # Check if the controller-manager is still running
  [[ -n "${CTLRMGR_PID-}" ]] && kube::util::read-array CTLRMGR_PIDS < <(pgrep -P "${CTLRMGR_PID}" ; ps -o pid= -p "${CTLRMGR_PID}")
  [[ -n "${CTLRMGR_PIDS-}" ]] && sudo kill "${CTLRMGR_PIDS[@]}" 2>/dev/null

  # Check if the cloud-controller-manager is still running
  [[ -n "${CLOUD_CTLRMGR_PID-}" ]] && kube::util::read-array CLOUD_CTLRMGR_PIDS < <(pgrep -P "${CLOUD_CTLRMGR_PID}" ; ps -o pid= -p "${CLOUD_CTLRMGR_PID}")
  [[ -n "${CLOUD_CTLRMGR_PIDS-}" ]] && sudo kill "${CLOUD_CTLRMGR_PIDS[@]}" 2>/dev/null

  # Check if the kubelet is still running
  [[ -n "${KUBELET_PID-}" ]] && kube::util::read-array KUBELET_PIDS < <(pgrep -P "${KUBELET_PID}" ; ps -o pid= -p "${KUBELET_PID}")
  [[ -n "${KUBELET_PIDS-}" ]] && sudo kill "${KUBELET_PIDS[@]}" 2>/dev/null

  # Check if the proxy is still running
  [[ -n "${PROXY_PID-}" ]] && kube::util::read-array PROXY_PIDS < <(pgrep -P "${PROXY_PID}" ; ps -o pid= -p "${PROXY_PID}")
  [[ -n "${PROXY_PIDS-}" ]] && sudo kill "${PROXY_PIDS[@]}" 2>/dev/null

  # Check if the scheduler is still running
  [[ -n "${SCHEDULER_PID-}" ]] && kube::util::read-array SCHEDULER_PIDS < <(pgrep -P "${SCHEDULER_PID}" ; ps -o pid= -p "${SCHEDULER_PID}")
  [[ -n "${SCHEDULER_PIDS-}" ]] && sudo kill "${SCHEDULER_PIDS[@]}" 2>/dev/null

  # Check if the etcd is still running
  [[ -n "${ETCD_PID-}" ]] && kube::etcd::stop
  if [[ "${PRESERVE_ETCD}" == "false" ]]; then
    [[ -n "${ETCD_DIR-}" ]] && kube::etcd::clean_etcd_dir
  fi

  # Cleanup dmesg running in the background
  [[ -n "${DMESG_PID-}" ]] && sudo kill "$DMESG_PID" 2>/dev/null

  exit 0
}

# Check if all processes are still running. Prints a warning once each time
# a process dies unexpectedly.
function healthcheck {
  if [[ -n "${APISERVER_PID-}" ]] && ! sudo kill -0 "${APISERVER_PID}" 2>/dev/null; then
    warning_log "API server terminated unexpectedly, see ${APISERVER_LOG}"
    APISERVER_PID=
  fi

  if [[ -n "${CTLRMGR_PID-}" ]] && ! sudo kill -0 "${CTLRMGR_PID}" 2>/dev/null; then
    warning_log "kube-controller-manager terminated unexpectedly, see ${CTLRMGR_LOG}"
    CTLRMGR_PID=
  fi

  if [[ -n "${KUBELET_PID-}" ]] && ! sudo kill -0 "${KUBELET_PID}" 2>/dev/null; then
    warning_log "kubelet terminated unexpectedly, see ${KUBELET_LOG}"
    KUBELET_PID=
  fi

  if [[ -n "${PROXY_PID-}" ]] && ! sudo kill -0 "${PROXY_PID}" 2>/dev/null; then
    warning_log "kube-proxy terminated unexpectedly, see ${PROXY_LOG}"
    PROXY_PID=
  fi

  if [[ -n "${SCHEDULER_PID-}" ]] && ! sudo kill -0 "${SCHEDULER_PID}" 2>/dev/null; then
    warning_log "scheduler terminated unexpectedly, see ${SCHEDULER_LOG}"
    SCHEDULER_PID=
  fi

  if [[ -n "${ETCD_PID-}" ]] && ! sudo kill -0 "${ETCD_PID}" 2>/dev/null; then
    warning_log "etcd terminated unexpectedly"
    ETCD_PID=
  fi
}

function print_color {
  message=$1
  prefix=${2:+$2: } # add colon only if defined
  color=${3:-1}     # default is red
  echo -n "$(tput bold)$(tput setaf "${color}")"
  echo "${prefix}${message}"
  echo -n "$(tput sgr0)"
}

function warning_log {
  print_color "$1" "W$(date "+%m%d %H:%M:%S")]" 1
}

function start_etcd {
    echo "Starting etcd"
    export ETCD_LOGFILE=${LOG_DIR}/etcd.log
    kube::etcd::start
}

function set_service_accounts {
    SERVICE_ACCOUNT_LOOKUP=${SERVICE_ACCOUNT_LOOKUP:-true}
    SERVICE_ACCOUNT_KEY=${SERVICE_ACCOUNT_KEY:-${TMP_DIR}/kube-serviceaccount.key}
    # Generate ServiceAccount key if needed
    if [[ ! -f "${SERVICE_ACCOUNT_KEY}" ]]; then
      mkdir -p "$(dirname "${SERVICE_ACCOUNT_KEY}")"
      openssl genrsa -out "${SERVICE_ACCOUNT_KEY}" 2048 2>/dev/null
    fi
}

function generate_certs {
    # Create CA signers
    if [[ "${ENABLE_SINGLE_CA_SIGNER:-}" = true ]]; then
        kube::util::create_signing_certkey "${CONTROLPLANE_SUDO}" "${CERT_DIR}" server '"client auth","server auth"'
        sudo cp "${CERT_DIR}/server-ca.key" "${CERT_DIR}/client-ca.key"
        sudo cp "${CERT_DIR}/server-ca.crt" "${CERT_DIR}/client-ca.crt"
        sudo cp "${CERT_DIR}/server-ca-config.json" "${CERT_DIR}/client-ca-config.json"
    else
        kube::util::create_signing_certkey "${CONTROLPLANE_SUDO}" "${CERT_DIR}" server '"server auth"'
        kube::util::create_signing_certkey "${CONTROLPLANE_SUDO}" "${CERT_DIR}" client '"client auth"'
    fi

    # Create auth proxy client ca
    kube::util::create_signing_certkey "${CONTROLPLANE_SUDO}" "${CERT_DIR}" request-header '"client auth"'

    # serving cert for kube-apiserver
    kube::util::create_serving_certkey "${CONTROLPLANE_SUDO}" "${CERT_DIR}" "server-ca" kube-apiserver kubernetes.default kubernetes.default.svc "localhost" "${API_HOST_IP}" "${API_HOST}" "${FIRST_SERVICE_CLUSTER_IP}"

    # Create client certs signed with client-ca, given id, given CN and a number of groups
    kube::util::create_client_certkey "${CONTROLPLANE_SUDO}" "${CERT_DIR}" 'client-ca' controller system:kube-controller-manager
    kube::util::create_client_certkey "${CONTROLPLANE_SUDO}" "${CERT_DIR}" 'client-ca' scheduler  system:kube-scheduler
    kube::util::create_client_certkey "${CONTROLPLANE_SUDO}" "${CERT_DIR}" 'client-ca' admin system:admin system:masters
    kube::util::create_client_certkey "${CONTROLPLANE_SUDO}" "${CERT_DIR}" 'client-ca' kube-apiserver kube-apiserver

    # Create matching certificates for kube-aggregator
    kube::util::create_serving_certkey "${CONTROLPLANE_SUDO}" "${CERT_DIR}" "server-ca" kube-aggregator api.kube-public.svc "localhost" "${API_HOST_IP}"
    kube::util::create_client_certkey "${CONTROLPLANE_SUDO}" "${CERT_DIR}" request-header-ca auth-proxy system:auth-proxy

    # TODO remove masters and add rolebinding
    kube::util::create_client_certkey "${CONTROLPLANE_SUDO}" "${CERT_DIR}" 'client-ca' kube-aggregator system:kube-aggregator system:masters
    kube::util::write_client_kubeconfig "${CONTROLPLANE_SUDO}" "${CERT_DIR}" "${ROOT_CA_FILE}" "${API_HOST}" "${API_SECURE_PORT}" kube-aggregator
}

function generate_kubeproxy_certs {
    kube::util::create_client_certkey "${CONTROLPLANE_SUDO}" "${CERT_DIR}" 'client-ca' kube-proxy system:kube-proxy system:nodes
    kube::util::write_client_kubeconfig "${CONTROLPLANE_SUDO}" "${CERT_DIR}" "${ROOT_CA_FILE}" "${API_HOST}" "${API_SECURE_PORT}" kube-proxy
}

function generate_kubelet_certs {
    kube::util::create_client_certkey "${CONTROLPLANE_SUDO}" "${CERT_DIR}" 'client-ca' kubelet "system:node:${HOSTNAME_OVERRIDE}" system:nodes
    kube::util::write_client_kubeconfig "${CONTROLPLANE_SUDO}" "${CERT_DIR}" "${ROOT_CA_FILE}" "${API_HOST}" "${API_SECURE_PORT}" kubelet
}

function start_apiserver {
    authorizer_args=()
    if [[ -n "${AUTHORIZATION_CONFIG:-}" ]]; then
      authorizer_args+=("--authorization-config=${AUTHORIZATION_CONFIG}")
    else
      if [[ -n "${AUTHORIZATION_MODE:-Node,RBAC}" ]]; then
        authorizer_args+=("--authorization-mode=${AUTHORIZATION_MODE:-Node,RBAC}")
      fi
      authorizer_args+=(
        "--authorization-webhook-config-file=${AUTHORIZATION_WEBHOOK_CONFIG_FILE}"
        "--authentication-token-webhook-config-file=${AUTHENTICATION_WEBHOOK_CONFIG_FILE}"
      )
    fi

    priv_arg=""
    if [[ -n "${ALLOW_PRIVILEGED}" ]]; then
      priv_arg="--allow-privileged=${ALLOW_PRIVILEGED}"
    fi

    runtime_config=""
    if [[ -n "${RUNTIME_CONFIG}" ]]; then
      runtime_config="--runtime-config=${RUNTIME_CONFIG}"
    fi

    # Let the API server pick a default address when API_HOST_IP
    # is set to 127.0.0.1
    advertise_address=""
    if [[ "${API_HOST_IP}" != "127.0.0.1" ]]; then
        advertise_address="--advertise-address=${API_HOST_IP}"
    fi
    if [[ "${ADVERTISE_ADDRESS}" != "" ]] ; then
        advertise_address="--advertise-address=${ADVERTISE_ADDRESS}"
    fi
    node_port_range=""
    if [[ "${NODE_PORT_RANGE}" != "" ]] ; then
        node_port_range="--service-node-port-range=${NODE_PORT_RANGE}"
    fi

    if [[ "${REUSE_CERTS}" != true ]]; then
      # Clean previous dynamic certs
      # This file is owned by root, so we can't always overwrite it (depends if
      # we run the script as root or not). Let's remove it, that is something we
      # can always do: either we have write permissions as a user in CERT_DIR or
      # we run the rm with sudo.
      ${CONTROLPLANE_SUDO} rm -f "${CERT_DIR}"/kubelet-rotated.kubeconfig

      # Create Certs
      generate_certs
    fi

    if [[ -z "${EGRESS_SELECTOR_CONFIG_FILE:-}" ]]; then
      cat <<EOF > "${TMP_DIR}"/kube_egress_selector_configuration.yaml
apiVersion: apiserver.k8s.io/v1beta1
kind: EgressSelectorConfiguration
egressSelections:
- name: cluster
  connection:
    proxyProtocol: Direct
- name: controlplane
  connection:
    proxyProtocol: Direct
- name: etcd
  connection:
    proxyProtocol: Direct
EOF
      EGRESS_SELECTOR_CONFIG_FILE="${TMP_DIR}/kube_egress_selector_configuration.yaml"
    fi

    if [[ -z "${AUDIT_POLICY_FILE}" ]]; then
      cat <<EOF > "${TMP_DIR}"/kube-audit-policy-file
# Log all requests at the Metadata level.
apiVersion: audit.k8s.io/v1
kind: Policy
rules:
- level: Metadata
EOF
      AUDIT_POLICY_FILE="${TMP_DIR}/kube-audit-policy-file"
    fi

    # Create admin config. Works without the apiserver, so do it early to enable debug access to the apiserver while it starts.
    kube::util::write_client_kubeconfig "${CONTROLPLANE_SUDO}" "${CERT_DIR}" "${ROOT_CA_FILE}" "${API_HOST}" "${API_SECURE_PORT}" admin
    ${CONTROLPLANE_SUDO} chown "${USER}" "${CERT_DIR}/client-admin.key" # make readable for kubectl

    APISERVER_LOG=${LOG_DIR}/kube-apiserver.log
    # shellcheck disable=SC2086
    run kube-apiserver "${APISERVER_LOG}" ${CONTROLPLANE_SUDO} "${GO_OUT}/kube-apiserver" "${authorizer_args[@]}" "${priv_arg}" ${runtime_config} \
      "${advertise_address}" \
      "${node_port_range}" \
      --v="${LOG_LEVEL}" \
      --vmodule="${LOG_SPEC}" \
      --audit-policy-file="${AUDIT_POLICY_FILE}" \
      --audit-log-path="${LOG_DIR}/kube-apiserver-audit.log" \
      --cert-dir="${CERT_DIR}" \
      --egress-selector-config-file="${EGRESS_SELECTOR_CONFIG_FILE:-}" \
      --client-ca-file="${CERT_DIR}/client-ca.crt" \
      --kubelet-client-certificate="${CERT_DIR}/client-kube-apiserver.crt" \
      --kubelet-certificate-authority="${CLUSTER_SIGNING_CERT_FILE}" \
      --kubelet-client-key="${CERT_DIR}/client-kube-apiserver.key" \
      --kubelet-port="${KUBELET_PORT}" \
      --kubelet-read-only-port="${KUBELET_READ_ONLY_PORT}" \
      --service-account-key-file="${SERVICE_ACCOUNT_KEY}" \
      --service-account-lookup="${SERVICE_ACCOUNT_LOOKUP}" \
      --service-account-issuer="https://kubernetes.default.svc" \
      --service-account-jwks-uri="https://kubernetes.default.svc/openid/v1/jwks" \
      --service-account-signing-key-file="${SERVICE_ACCOUNT_KEY}" \
      --enable-admission-plugins="${ENABLE_ADMISSION_PLUGINS}" \
      --disable-admission-plugins="${DISABLE_ADMISSION_PLUGINS}" \
      --admission-control-config-file="${ADMISSION_CONTROL_CONFIG_FILE}" \
      --bind-address="${API_BIND_ADDR}" \
      --secure-port="${API_SECURE_PORT}" \
      --tls-cert-file="${CERT_DIR}/serving-kube-apiserver.crt" \
      --tls-private-key-file="${CERT_DIR}/serving-kube-apiserver.key" \
      --storage-backend="${STORAGE_BACKEND}" \
      --storage-media-type="${STORAGE_MEDIA_TYPE}" \
      --etcd-servers="http://${ETCD_HOST}:${ETCD_PORT}" \
      --service-cluster-ip-range="${SERVICE_CLUSTER_IP_RANGE}" \
      --feature-gates="${FEATURE_GATES}" \
      --emulated-version="${EMULATED_VERSION}" \
      --external-hostname="${EXTERNAL_HOSTNAME}" \
      --requestheader-username-headers=X-Remote-User \
      --requestheader-group-headers=X-Remote-Group \
      --requestheader-extra-headers-prefix=X-Remote-Extra- \
      --requestheader-client-ca-file="${CERT_DIR}/request-header-ca.crt" \
      --requestheader-allowed-names=system:auth-proxy \
      --proxy-client-cert-file="${CERT_DIR}/client-auth-proxy.crt" \
      --proxy-client-key-file="${CERT_DIR}/client-auth-proxy.key" \
      --cors-allowed-origins="${API_CORS_ALLOWED_ORIGINS}" &
    APISERVER_PID=$!

    if [[ -z "${DRY_RUN}" ]]; then
        # Wait for kube-apiserver to come up before launching the rest of the components.
        echo "Waiting for apiserver to come up"
        kube::util::wait_for_url "https://${API_HOST_IP}:${API_SECURE_PORT}/healthz" "apiserver: " 1 "${WAIT_FOR_URL_API_SERVER}" "${MAX_TIME_FOR_URL_API_SERVER}" \
            || { echo "check apiserver logs: ${APISERVER_LOG}" ; exit 1 ; }
    fi

    # Create kubeconfigs for all components, using client certs. This needs a running apiserver.
    kube::util::write_client_kubeconfig "${CONTROLPLANE_SUDO}" "${CERT_DIR}" "${ROOT_CA_FILE}" "${API_HOST}" "${API_SECURE_PORT}" controller
    kube::util::write_client_kubeconfig "${CONTROLPLANE_SUDO}" "${CERT_DIR}" "${ROOT_CA_FILE}" "${API_HOST}" "${API_SECURE_PORT}" scheduler

    if [[ -z "${AUTH_ARGS}" ]]; then
        AUTH_ARGS="--client-key=${CERT_DIR}/client-admin.key --client-certificate=${CERT_DIR}/client-admin.crt"
    fi

    # Grant apiserver permission to speak to the kubelet
    run kubectl "" "${KUBECTL}" --kubeconfig "${CERT_DIR}/admin.kubeconfig" create clusterrolebinding kube-apiserver-kubelet-admin --clusterrole=system:kubelet-api-admin --user=kube-apiserver

    # Grant kubelets permission to request client certificates
    run kubectl "" "${KUBECTL}" --kubeconfig "${CERT_DIR}/admin.kubeconfig" create clusterrolebinding kubelet-csr --clusterrole=system:certificates.k8s.io:certificatesigningrequests:selfnodeclient --group=system:nodes

    ${CONTROLPLANE_SUDO} cp "${CERT_DIR}/admin.kubeconfig" "${CERT_DIR}/admin-kube-aggregator.kubeconfig"
    ${CONTROLPLANE_SUDO} chown -R "$(whoami)" "${CERT_DIR}"
    run kubectl "" "${KUBECTL}" config set-cluster local-up-cluster --kubeconfig="${CERT_DIR}/admin-kube-aggregator.kubeconfig" --server="https://${API_HOST_IP}:31090"
    echo "use 'kubectl --kubeconfig=${CERT_DIR}/admin-kube-aggregator.kubeconfig' to use the aggregated API server"

}

function start_controller_manager {
    cloud_config_arg=("--cloud-provider=${CLOUD_PROVIDER}" "--cloud-config=${CLOUD_CONFIG}")
    cloud_config_arg+=("--configure-cloud-routes=${CONFIGURE_CLOUD_ROUTES}")
    if [[ "${EXTERNAL_CLOUD_PROVIDER:-}" == "true" ]]; then
      cloud_config_arg=("--cloud-provider=external")
      cloud_config_arg+=("--external-cloud-volume-plugin=${EXTERNAL_CLOUD_VOLUME_PLUGIN}")
      cloud_config_arg+=("--cloud-config=${CLOUD_CONFIG}")
    fi

    CTLRMGR_LOG=${LOG_DIR}/kube-controller-manager.log
    # shellcheck disable=SC2086
    run kube-controller-manager "${CTLRMGR_LOG}" ${CONTROLPLANE_SUDO} "${GO_OUT}/kube-controller-manager" \
      --v="${LOG_LEVEL}" \
      --vmodule="${LOG_SPEC}" \
      --service-account-private-key-file="${SERVICE_ACCOUNT_KEY}" \
      --service-cluster-ip-range="${SERVICE_CLUSTER_IP_RANGE}" \
      --root-ca-file="${ROOT_CA_FILE}" \
      --cluster-signing-cert-file="${CLUSTER_SIGNING_CERT_FILE}" \
      --cluster-signing-key-file="${CLUSTER_SIGNING_KEY_FILE}" \
      --enable-hostpath-provisioner="${ENABLE_HOSTPATH_PROVISIONER}" \
      --pvclaimbinder-sync-period="${CLAIM_BINDER_SYNC_PERIOD}" \
      --feature-gates="${FEATURE_GATES}" \
      --emulated-version="${EMULATED_VERSION}" \
      "${cloud_config_arg[@]}" \
      --authentication-kubeconfig "${CERT_DIR}"/controller.kubeconfig \
      --authorization-kubeconfig "${CERT_DIR}"/controller.kubeconfig \
      --kubeconfig "${CERT_DIR}"/controller.kubeconfig \
      --use-service-account-credentials \
      --controllers="${KUBE_CONTROLLERS}" \
      --leader-elect="${LEADER_ELECT}" \
      --cert-dir="${CERT_DIR}" \
      --secure-port="${KCM_SECURE_PORT}" \
      --bind-address="${API_BIND_ADDR}" \
      --master="https://${API_HOST}:${API_SECURE_PORT}" &
    CTLRMGR_PID=$!
}

function start_cloud_controller_manager {
    if [ -z "${CLOUD_CONFIG}" ]; then
      echo "CLOUD_CONFIG cannot be empty!"
      exit 1
    fi
    if [ ! -f "${CLOUD_CONFIG}" ]; then
      echo "Cloud config ${CLOUD_CONFIG} doesn't exist"
      exit 1
    fi

    CLOUD_CTLRMGR_LOG=${LOG_DIR}/cloud-controller-manager.log
    # shellcheck disable=SC2086
    run cloud-controller-manager ${CONTROLPLANE_SUDO} "${EXTERNAL_CLOUD_PROVIDER_BINARY:-"${GO_OUT}/cloud-controller-manager"}" \
      ${CLOUD_CTLRMGR_FLAGS} \
      --v="${LOG_LEVEL}" \
      --vmodule="${LOG_SPEC}" \
      --feature-gates="${FEATURE_GATES}" \
      --cloud-provider="${CLOUD_PROVIDER}" \
      --cloud-config="${CLOUD_CONFIG}" \
      --configure-cloud-routes="${CONFIGURE_CLOUD_ROUTES}" \
      --kubeconfig "${CERT_DIR}"/controller.kubeconfig \
      --use-service-account-credentials \
      --leader-elect="${LEADER_ELECT}" \
      --master="https://${API_HOST}:${API_SECURE_PORT}" >"${CLOUD_CTLRMGR_LOG}" 2>&1 &
    export CLOUD_CTLRMGR_PID=$!
}

function wait_node_csr() {
  local interval_time=2
  local csr_approved_time=300
  local newline='"\n"'
  local unapproved_csr_names="--field-selector='spec.signerName=kubernetes.io/kubelet-serving' -o go-template='{{range .items}}{{if not .status}}{{.metadata.name}}{{${newline}}}{{end}}{{end}}"
  local csr_approved="${KUBECTL} --kubeconfig '${CERT_DIR}/admin.kubeconfig' get csr ${unapproved_csr_names}' | xargs --no-run-if-empty ${KUBECTL} --kubeconfig '${CERT_DIR}/admin.kubeconfig' certificate approve | grep csr"
  kube::util::wait_for_success "$csr_approved_time" "$interval_time" "$csr_approved"
  if [ $? == "1" ]; then
    echo "time out on waiting for CSR approval"
    exit 1
  fi
  echo "kubelet CSR approved"
}

function wait_node_ready(){
  if [[ -n "${DRY_RUN}" ]]; then
    return
  fi

  echo "wait kubelet ready"

  # check the nodes information after kubelet daemon start
  local nodes_stats="${KUBECTL} --kubeconfig '${CERT_DIR}/admin.kubeconfig' get nodes"
  local node_name=$HOSTNAME_OVERRIDE
  local system_node_wait_time=60
  local interval_time=2
  kube::util::wait_for_success "$system_node_wait_time" "$interval_time" "$nodes_stats | grep $node_name"
  if [ $? == "1" ]; then
    echo "time out on waiting $node_name exist"
    exit 1
  fi

  local system_node_ready_time=300
  local node_ready="${KUBECTL} --kubeconfig '${CERT_DIR}/admin.kubeconfig' wait --for=condition=Ready --timeout=60s nodes $node_name"
  kube::util::wait_for_success "$system_node_ready_time" "$interval_time" "$node_ready"
  if [ $? == "1" ]; then
    echo "time out on waiting $node_name info"
    exit 1
  fi
}

function wait_coredns_available(){
  if [[ -n "${DRY_RUN}" ]]; then
    return
  fi

  local interval_time=2
  local coredns_wait_time=300

  # kick the coredns pods to be recreated
  ${KUBECTL} --kubeconfig="${CERT_DIR}/admin.kubeconfig" -n kube-system delete pods -l k8s-app=kube-dns
  sleep 30

  local coredns_pods_ready="${KUBECTL} --kubeconfig '${CERT_DIR}/admin.kubeconfig' wait --for=condition=Ready --timeout=60s pods -l k8s-app=kube-dns -n kube-system"
  kube::util::wait_for_success "$coredns_wait_time" "$interval_time" "$coredns_pods_ready"
  if [ $? == "1" ]; then
    echo "time out on waiting for coredns pods"
    exit 1
  fi

  local coredns_available="${KUBECTL} --kubeconfig '${CERT_DIR}/admin.kubeconfig' wait --for=condition=Available --timeout=60s deployments coredns -n kube-system"
  kube::util::wait_for_success "$coredns_wait_time" "$interval_time" "$coredns_available"
  if [ $? == "1" ]; then
    echo "time out on waiting for coredns deployment"
    exit 1
  fi

  if [[ "${ENABLE_DAEMON}" = false ]]; then
    # bump log level
    echo "6" | sudo tee /proc/sys/kernel/printk

    # loop through and grab all things in dmesg
    # shellcheck disable=SC2024
    sudo dmesg > "${LOG_DIR}/dmesg.log"
    # shellcheck disable=SC2024
    sudo dmesg -w --human >> "${LOG_DIR}/dmesg.log" &
    DMESG_PID=$!
  fi
}

function start_kubelet {
    KUBELET_LOG=${LOG_DIR}/kubelet.log
    mkdir -p "${POD_MANIFEST_PATH}" &>/dev/null || sudo mkdir -p "${POD_MANIFEST_PATH}"

    cloud_config_arg=("--cloud-provider=${CLOUD_PROVIDER}")
    if [[ "${EXTERNAL_CLOUD_PROVIDER:-}" == "true" ]]; then
       cloud_config_arg=("--cloud-provider=external")
       if [[ "${CLOUD_PROVIDER:-}" == "aws" ]]; then
         cloud_config_arg+=("--provider-id=$(curl http://169.254.169.254/latest/meta-data/instance-id)")
       else
         cloud_config_arg+=("--provider-id=${KUBELET_PROVIDER_ID}")
       fi
    fi

    mkdir -p "/var/lib/kubelet" &>/dev/null || sudo mkdir -p "/var/lib/kubelet"

    image_service_endpoint_args=()
    if [[ -n "${IMAGE_SERVICE_ENDPOINT}" ]]; then
      image_service_endpoint_args=("--image-service-endpoint=${IMAGE_SERVICE_ENDPOINT}")
    fi

    # shellcheck disable=SC2206
    all_kubelet_flags=(
      "--v=${LOG_LEVEL}"
      "--vmodule=${LOG_SPEC}"
      "--hostname-override=${HOSTNAME_OVERRIDE}"
      "${cloud_config_arg[@]}"
      "--bootstrap-kubeconfig=${CERT_DIR}/kubelet.kubeconfig"
      "--kubeconfig=${CERT_DIR}/kubelet-rotated.kubeconfig"
      ${image_service_endpoint_args[@]+"${image_service_endpoint_args[@]}"}
      ${KUBELET_FLAGS}
    )

    # warn if users are running with swap allowed
    if [ "${FAIL_SWAP_ON}" == "false" ]; then
        echo "WARNING : The kubelet is configured to not fail even if swap is enabled; production deployments should disable swap unless testing NodeSwap feature."
    fi

    if [[ "${REUSE_CERTS}" != true ]]; then
        # clear previous dynamic certs
        sudo rm -fr "/var/lib/kubelet/pki" "${CERT_DIR}/kubelet-rotated.kubeconfig"
        # create new certs
        generate_kubelet_certs
    fi

    cat <<EOF > "${TMP_DIR}"/kubelet.yaml
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
address: "${KUBELET_HOST}"
cgroupDriver: "${CGROUP_DRIVER}"
cgroupRoot: "${CGROUP_ROOT}"
cgroupsPerQOS: ${CGROUPS_PER_QOS}
containerRuntimeEndpoint: ${CONTAINER_RUNTIME_ENDPOINT}
cpuCFSQuota: ${CPU_CFS_QUOTA}
enableControllerAttachDetach: ${ENABLE_CONTROLLER_ATTACH_DETACH}
localStorageCapacityIsolation: ${LOCAL_STORAGE_CAPACITY_ISOLATION}
evictionPressureTransitionPeriod: "${EVICTION_PRESSURE_TRANSITION_PERIOD}"
failSwapOn: ${FAIL_SWAP_ON}
port: ${KUBELET_PORT}
readOnlyPort: ${KUBELET_READ_ONLY_PORT}
healthzPort: ${KUBELET_HEALTHZ_PORT}
healthzBindAddress: ${KUBELET_HOST}
rotateCertificates: true
serverTLSBootstrap: true
runtimeRequestTimeout: "${RUNTIME_REQUEST_TIMEOUT}"
staticPodPath: "${POD_MANIFEST_PATH}"
resolvConf: "${KUBELET_RESOLV_CONF}"
EOF

  if [[ "$ENABLE_TRACING" = true ]]; then
        cat <<EOF >> "${TMP_DIR}"/kubelet.yaml
tracing:
  endpoint: localhost:4317 # the default value
  samplingRatePerMillion: 1000000 # sample always
EOF
    fi

    if [[ "$LIMITED_SWAP" == "true" ]]; then
        cat <<EOF >> "${TMP_DIR}"/kubelet.yaml
memorySwap:
  swapBehavior: LimitedSwap
EOF
    fi

    {
      # authentication
      echo "authentication:"
      echo "  webhook:"
      if [[ "${KUBELET_AUTHENTICATION_WEBHOOK:-}" != "false" ]]; then
        echo "    enabled: true"
      else
        echo "    enabled: false"
      fi
      echo "  x509:"
      if [[ -n "${CLIENT_CA_FILE:-}" ]]; then
        echo "    clientCAFile: \"${CLIENT_CA_FILE}\""
      else
        echo "    clientCAFile: \"${CERT_DIR}/client-ca.crt\""
      fi

      # authorization
      if [[ "${KUBELET_AUTHORIZATION_WEBHOOK:-}" != "false" ]]; then
        echo "authorization:"
        echo "  mode: Webhook"
      fi

      # dns
      if [[ "${ENABLE_CLUSTER_DNS}" = true ]]; then
        if [[ "${ENABLE_NODELOCAL_DNS:-}" == "true" ]]; then
          echo "clusterDNS: [ \"${LOCAL_DNS_IP}\" ]"
        else
          echo "clusterDNS: [ \"${DNS_SERVER_IP}\" ]"
        fi
        echo "clusterDomain: \"${DNS_DOMAIN}\""
      else
        # To start a private DNS server set ENABLE_CLUSTER_DNS and
        # DNS_SERVER_IP/DOMAIN. This will at least provide a working
        # DNS server for real world hostnames.
        echo "clusterDNS: [ \"8.8.8.8\" ]"
      fi

      # eviction
      if [[ -n ${EVICTION_HARD} ]]; then
        echo "evictionHard:"
        parse_eviction "${EVICTION_HARD}"
      fi
      if [[ -n ${EVICTION_SOFT} ]]; then
        echo "evictionSoft:"
        parse_eviction "${EVICTION_SOFT}"
      fi

      # feature gate
      if [[ -n ${FEATURE_GATES} ]]; then
        parse_feature_gates "${FEATURE_GATES}"
      fi

      # topology maanager policy
      if [[ -n ${TOPOLOGY_MANAGER_POLICY} ]]; then
        echo "topologyManagerPolicy: \"${TOPOLOGY_MANAGER_POLICY}\""
      fi

      # cpumanager policy
      if [[ -n ${CPUMANAGER_POLICY} ]]; then
        echo "cpuManagerPolicy: \"${CPUMANAGER_POLICY}\""
      fi

      # cpumanager reconcile period
      if [[ -n ${CPUMANAGER_RECONCILE_PERIOD} ]]; then
	echo "cpuManagerReconcilePeriod: \"${CPUMANAGER_RECONCILE_PERIOD}\""
      fi

      # cpumanager policy options
      if [[ -n ${CPUMANAGER_POLICY_OPTIONS} ]]; then
	parse_cpumanager_policy_options "${CPUMANAGER_POLICY_OPTIONS}"
      fi

    } >>"${TMP_DIR}"/kubelet.yaml

    # shellcheck disable=SC2024
    run kubelet "${KUBELET_LOG}" sudo -E "${GO_OUT}/kubelet" "${all_kubelet_flags[@]}" \
      --config="${TMP_DIR}"/kubelet.yaml &
    KUBELET_PID=$!

    # Quick check that kubelet is running.
    if [ -n "${DRY_RUN}" ] || ( [ -n "${KUBELET_PID}" ] && ps -p ${KUBELET_PID} > /dev/null ); then
      echo "kubelet ( ${KUBELET_PID} ) is running."
    else
      cat "${KUBELET_LOG}" ; exit 1
    fi
}

function start_kubeproxy {
    PROXY_LOG=${LOG_DIR}/kube-proxy.log

    if [[ "${START_MODE}" != *"nokubelet"* ]]; then
      # wait for kubelet collect node information
      wait_node_ready
    fi

    cat <<EOF > "${TMP_DIR}"/kube-proxy.yaml
apiVersion: kubeproxy.config.k8s.io/v1alpha1
kind: KubeProxyConfiguration
clientConnection:
  kubeconfig: ${CERT_DIR}/kube-proxy.kubeconfig
hostnameOverride: ${HOSTNAME_OVERRIDE}
mode: ${KUBE_PROXY_MODE}
conntrack:
# Skip setting sysctl value "net.netfilter.nf_conntrack_max"
  maxPerCore: 0
# Skip setting "net.netfilter.nf_conntrack_tcp_timeout_established"
  tcpEstablishedTimeout: 0s
# Skip setting "net.netfilter.nf_conntrack_tcp_timeout_close"
  tcpCloseWaitTimeout: 0s
EOF
    if [[ -n ${FEATURE_GATES} ]]; then
      parse_feature_gates "${FEATURE_GATES}"
    fi >>"${TMP_DIR}"/kube-proxy.yaml

    if [[ "${REUSE_CERTS}" != true ]]; then
        generate_kubeproxy_certs
    fi

    # Including the port in the bind addresses mirrors the defaults.
    # Probably not necessary...
    #
    # shellcheck disable=SC2024
    run kube-proxy "${PROXY_LOG}" sudo "${GO_OUT}/kube-proxy" \
      --v="${LOG_LEVEL}" \
      --config="${TMP_DIR}"/kube-proxy.yaml \
      --healthz-port="${PROXY_HEALTHZ_PORT}" \
      --healthz-bind-address="${API_BIND_ADDR}:${PROXY_HEALTHZ_PORT}" \
      --metrics-port="${PROXY_METRICS_PORT}" \
      --metrics-bind-address="${API_BIND_ADDR}:${PROXY_METRICS_PORT}" \
      --master="https://${API_HOST}:${API_SECURE_PORT}" &
    PROXY_PID=$!
}

function start_kubescheduler {
    SCHEDULER_LOG=${LOG_DIR}/kube-scheduler.log

    cat <<EOF > "${TMP_DIR}"/kube-scheduler.yaml
apiVersion: kubescheduler.config.k8s.io/v1
kind: KubeSchedulerConfiguration
clientConnection:
  kubeconfig: ${CERT_DIR}/scheduler.kubeconfig
leaderElection:
  leaderElect: ${LEADER_ELECT}
EOF
    # shellcheck disable=SC2086
    run kube-scheduler "${SCHEDULER_LOG}" ${CONTROLPLANE_SUDO} "${GO_OUT}/kube-scheduler" \
      --v="${LOG_LEVEL}" \
      --config="${TMP_DIR}"/kube-scheduler.yaml \
      --feature-gates="${FEATURE_GATES}" \
      --emulated-version="${EMULATED_VERSION}" \
      --authentication-kubeconfig "${CERT_DIR}"/scheduler.kubeconfig \
      --authorization-kubeconfig "${CERT_DIR}"/scheduler.kubeconfig \
      --secure-port="${SCHEDULER_SECURE_PORT}" \
      --bind-address="${API_BIND_ADDR}" \
      --master="https://${API_HOST}:${API_SECURE_PORT}" &
    SCHEDULER_PID=$!
}

function start_dns_addon {
    if [[ "${ENABLE_CLUSTER_DNS}" = true ]]; then
        cp "${KUBE_ROOT}/cluster/addons/dns/${DNS_ADDON}/${DNS_ADDON}.yaml.in" "${TMP_DIR}/dns.yaml"
        ${SED} -i -e "s/dns_domain/${DNS_DOMAIN}/g" "${TMP_DIR}/dns.yaml"
        ${SED} -i -e "s/dns_server/${DNS_SERVER_IP}/g" "${TMP_DIR}/dns.yaml"
        ${SED} -i -e "s/dns_memory_limit/${DNS_MEMORY_LIMIT}/g" "${TMP_DIR}/dns.yaml"
        # TODO update to dns role once we have one.
        # use kubectl to create dns addon
        if run kubectl "" "${KUBECTL}" --kubeconfig="${CERT_DIR}/admin.kubeconfig" --namespace=kube-system apply -f "${TMP_DIR}/dns.yaml" ; then
            echo "${DNS_ADDON} addon successfully deployed."
        else
		echo "Something is wrong with your DNS input"
		cat "${TMP_DIR}/dns.yaml"
		exit 1
        fi
    fi
}

function start_nodelocaldns {
  cp "${KUBE_ROOT}/cluster/addons/dns/nodelocaldns/nodelocaldns.yaml" "${TMP_DIR}/nodelocaldns.yaml"
  # eventually all the __PILLAR__ stuff will be gone, but theyre still in nodelocaldns for backward compat.
  ${SED} -i -e "s/__PILLAR__DNS__DOMAIN__/${DNS_DOMAIN}/g" "${TMP_DIR}/nodelocaldns.yaml"
  ${SED} -i -e "s/__PILLAR__DNS__SERVER__/${DNS_SERVER_IP}/g" "${TMP_DIR}/nodelocaldns.yaml"
  ${SED} -i -e "s/__PILLAR__LOCAL__DNS__/${LOCAL_DNS_IP}/g" "${TMP_DIR}/nodelocaldns.yaml"

  # use kubectl to create nodelocaldns addon
  run kubectl "" "${KUBECTL}" --kubeconfig="${CERT_DIR}/admin.kubeconfig" --namespace=kube-system apply -f "${TMP_DIR}/nodelocaldns.yaml"
  echo "NodeLocalDNS addon successfully deployed."
}

function start_csi_snapshotter {
    if [[ "${ENABLE_CSI_SNAPSHOTTER}" = true ]]; then
        echo "Creating Kubernetes-CSI snapshotter"
        run kubectl "" "${KUBECTL}" --kubeconfig="${CERT_DIR}/admin.kubeconfig" apply -f "${KUBE_ROOT}/cluster/addons/volumesnapshots/crd/snapshot.storage.k8s.io_volumesnapshots.yaml"
        run kubectl "" "${KUBECTL}" --kubeconfig="${CERT_DIR}/admin.kubeconfig" apply -f "${KUBE_ROOT}/cluster/addons/volumesnapshots/crd/snapshot.storage.k8s.io_volumesnapshotclasses.yaml"
        run kubectl "" "${KUBECTL}" --kubeconfig="${CERT_DIR}/admin.kubeconfig" apply -f "${KUBE_ROOT}/cluster/addons/volumesnapshots/crd/snapshot.storage.k8s.io_volumesnapshotcontents.yaml"
        run kubectl "" "${KUBECTL}" --kubeconfig="${CERT_DIR}/admin.kubeconfig" apply -f "${KUBE_ROOT}/cluster/addons/volumesnapshots/volume-snapshot-controller/rbac-volume-snapshot-controller.yaml"
        run kubectl "" "${KUBECTL}" --kubeconfig="${CERT_DIR}/admin.kubeconfig" apply -f "${KUBE_ROOT}/cluster/addons/volumesnapshots/volume-snapshot-controller/volume-snapshot-controller-deployment.yaml"

        echo "Kubernetes-CSI snapshotter successfully deployed."
    fi
}

function create_storage_class {
    if [ -z "${CLOUD_PROVIDER}" ]; then
        CLASS_FILE=${KUBE_ROOT}/cluster/addons/storage-class/local/default.yaml
    else
        CLASS_FILE=${KUBE_ROOT}/cluster/addons/storage-class/${CLOUD_PROVIDER}/default.yaml
    fi

    if [ -e "${CLASS_FILE}" ]; then
        echo "Create default storage class for ${CLOUD_PROVIDER}"
        run kubectl "" "${KUBECTL}" --kubeconfig="${CERT_DIR}/admin.kubeconfig" apply -f "${CLASS_FILE}"
    else
        echo "No storage class available for ${CLOUD_PROVIDER}."
    fi
}

function print_success {
if [[ -n "${DRY_RUN}" ]]; then
  return
fi

if [[ "${START_MODE}" != "kubeletonly" ]]; then
  if [[ "${ENABLE_DAEMON}" = false ]]; then
    echo "Local Kubernetes cluster is running. Press Ctrl-C to shut it down."
  else
    echo "Local Kubernetes cluster is running."
  fi

  echo
  echo "Configurations:"
  for f in "${TMP_DIR}"/*; do
    echo "  ${f}"
  done

  cat <<EOF

Logs:
  ${ETCD_LOGFILE:-}
  ${APISERVER_LOG:-}
  ${CTLRMGR_LOG:-}
  ${CLOUD_CTLRMGR_LOG:-}
  ${PROXY_LOG:-}
  ${SCHEDULER_LOG:-}
EOF
fi

if [[ "${START_MODE}" == "all" ]]; then
  echo "  ${KUBELET_LOG}"
elif [[ "${START_MODE}" == *"nokubelet"* ]]; then
  echo
  echo "No kubelet was started because you set START_MODE=nokubelet"
  echo "Run this script again with START_MODE=kubeletonly to run a kubelet"
fi

if [[ "${START_MODE}" != "kubeletonly" ]]; then
  echo
  if [[ "${ENABLE_DAEMON}" = false ]]; then
    echo "To start using your cluster, you can open up another terminal/tab and run:"
  else
    echo "To start using your cluster, run:"
  fi
  cat <<EOF

  export KUBECONFIG=${CERT_DIR}/admin.kubeconfig
  cluster/kubectl.sh

Alternatively, you can write to the default kubeconfig:

  export KUBERNETES_PROVIDER=local

  cluster/kubectl.sh config set-cluster local --server=https://${API_HOST}:${API_SECURE_PORT} --certificate-authority=${ROOT_CA_FILE}
  cluster/kubectl.sh config set-credentials myself ${AUTH_ARGS}
  cluster/kubectl.sh config set-context local --cluster=local --user=myself
  cluster/kubectl.sh config use-context local
  cluster/kubectl.sh
EOF
else
  cat <<EOF
The kubelet was started.

Logs:
  ${KUBELET_LOG}
EOF
fi
}

function parse_cpumanager_policy_options {
  echo "cpuManagerPolicyOptions:"
  # Convert from foo=true,bar=false to
  #   foo: "true"
  #   bar: "false"
  for option in $(echo "$1" | tr ',' ' '); do
    echo "${option}" | ${SED} -e 's/\(.*\)=\(.*\)/  \1: "\2"/'
  done
}

function parse_feature_gates {
  echo "featureGates:"
  # Convert from foo=true,bar=false to
  #   foo: true
  #   bar: false
  for gate in $(echo "$1" | tr ',' ' '); do
    echo "${gate}" | ${SED} -e 's/\(.*\)=\(.*\)/  \1: \2/'
  done
}

function parse_eviction {
  # Convert from memory.available<100Mi,nodefs.available<10%,nodefs.inodesFree<5% to
  #   memory.available: "100Mi"
  #   nodefs.available: "10%"
  #   nodefs.inodesFree: "5%"
  for eviction in $(echo "$1" | tr ',' ' '); do
    echo "${eviction}" | ${SED} -e 's/</: \"/' | ${SED} -e 's/^/  /' | ${SED} -e 's/$/\"/'
  done
}

function tolerate_cgroups_v2 {
  # https://github.com/moby/moby/blob/be220af9fb36e9baa9a75bbc41f784260aa6f96e/hack/dind#L28-L38
  # cgroup v2: enable nesting
  if [ -f /sys/fs/cgroup/cgroup.controllers ]; then
    # move the processes from the root group to the /init group,
    # otherwise writing subtree_control fails with EBUSY.
    # An error during moving non-existent process (i.e., "cat") is ignored.
    mkdir -p /sys/fs/cgroup/init
    xargs -rn1 < /sys/fs/cgroup/cgroup.procs > /sys/fs/cgroup/init/cgroup.procs || :
    # enable controllers
    sed -e 's/ / +/g' -e 's/^/+/' < /sys/fs/cgroup/cgroup.controllers \
      > /sys/fs/cgroup/cgroup.subtree_control
  fi
}

function install_cni {
  if [[ -n "${CNI_TARGETARCH}" ]]; then
    host_arch="${CNI_TARGETARCH}"
  else
    host_arch=$(detect_arch)
  fi

  cni_plugin_tarball="cni-plugins-linux-${host_arch}-${CNI_PLUGINS_VERSION}.tgz"
  cni_plugins_url="${CNI_PLUGINS_URL}/${CNI_PLUGINS_VERSION}/${cni_plugin_tarball}"
  cni_plugin_sha_url="${cni_plugins_url}.sha256"

  echo "Installing CNI plugin binaries ..." &&
    cd "${TMP_DIR}" &&
    curl -sSL --retry 5 -o "${cni_plugin_tarball}" "${cni_plugins_url}" &&
    curl -sSL --retry 5 -o "${cni_plugin_tarball}.sha256" "${cni_plugin_sha_url}" &&
    sha256sum -c "${cni_plugin_tarball}.sha256" &&
    rm -f "${cni_plugin_tarball}.sha256" &&
    sudo mkdir -p /opt/cni/bin &&
    sudo tar -C /opt/cni/bin -xzvf "${cni_plugin_tarball}" &&
    rm -rf "${cni_plugin_tarball}" &&
    sudo find /opt/cni/bin -type f -not \( \
        -iname host-local \
        -o -iname bridge \
        -o -iname portmap \
        -o -iname loopback \
        \) \
        -delete

  # containerd in kubekins supports CNI version 0.4.0
  echo "Configuring cni"
  sudo mkdir -p "$CNI_CONFIG_DIR"
  cat << EOF | sudo tee "$CNI_CONFIG_DIR"/10-containerd-net.conflist
{
 "cniVersion": "1.0.0",
 "name": "containerd-net",
 "plugins": [
   {
     "type": "bridge",
     "bridge": "cni0",
     "isGateway": true,
     "ipMasq": true,
     "promiscMode": true,
     "ipam": {
       "type": "host-local",
       "ranges": [
         [{
           "subnet": "10.88.0.0/16"
         }],
         [{
           "subnet": "2001:db8:4860::/64"
         }]
       ],
       "routes": [
         { "dst": "0.0.0.0/0" },
         { "dst": "::/0" }
       ]
     }
   },
   {
     "type": "portmap",
     "capabilities": {"portMappings": true},
     "externalSetMarkChain": "KUBE-MARK-MASQ"
   }
 ]
}
EOF
}

function install_cni_if_needed {
  echo "Checking CNI Installation at /opt/cni/bin"
  if ! command -v /opt/cni/bin/loopback &> /dev/null ; then
    echo "CNI Installation not found at /opt/cni/bin"
    install_cni
  fi
}

# If we are running in the CI, we need a few more things before we can start
if [[ "${KUBETEST_IN_DOCKER:-}" == "true" ]]; then
  echo "Preparing to test ..."
  "${KUBE_ROOT}"/hack/install-etcd.sh
  export PATH="${KUBE_ROOT}/third_party/etcd:${PATH}"
  KUBE_FASTBUILD=true make ginkgo cross

  log_dir="${LOG_DIR}"
  if [[ -n "${ARTIFACTS}" ]]; then
    # Store logs in the artifacts directory so that they are
    # available in the prow web UI.
    log_dir="${ARTIFACTS}"
  fi

  echo "install additional packages"
  apt-get update
  apt-get install -y conntrack dnsutils nftables ripgrep sudo tree vim

  # configure shared mounts to prevent failure in DIND scenarios
  mount --make-rshared /

  # to use containerd as kubelet container runtime we need to install cni
  install_cni 

  # If we are running in a cgroups v2 environment
  # we need to enable nesting
  tolerate_cgroups_v2

  echo "stopping docker"
  service docker stop

  # output version info
  containerd --version
  runc --version

  # configure and start containerd
  echo "configuring containerd"
  containerd config default > /etc/containerd/config.toml
  sed -ie 's|root = ./var/lib/containerd.|root = "/docker-graph/containerd/daemon"|' /etc/containerd/config.toml
  sed -ie 's|state = ./run/containerd.|state = "/var/run/docker/containerd/daemon"|' /etc/containerd/config.toml
  sed -ie 's|enable_cdi = false|enable_cdi = true|' /etc/containerd/config.toml

  echo "starting containerd"
  containerd_endpoint="${CONTAINER_RUNTIME_ENDPOINT#unix://}"
  start-stop-daemon --start --background --pidfile /var/run/containerd.pid --output "${log_dir}/containerd.log" \
          --exec /usr/bin/containerd -- --address "${containerd_endpoint}"
  kube::util::wait_for_success 60 2 "ctr --address ${containerd_endpoint} images list"
  if [ $? == "1" ]; then
    echo "time out on waiting for containerd to start"
    exit 1
  fi
fi

# validate that etcd is: not running, in path, and has minimum required version.
if [[ "${START_MODE}" != "kubeletonly" ]]; then
  kube::etcd::validate
fi

if [[ "${START_MODE}" != "kubeletonly" ]]; then
  test_apiserver_off
fi

kube::util::test_openssl_installed
kube::util::ensure-cfssl

### IF the user didn't supply an output/ for the build... Then we detect.
if [ "${GO_OUT}" == "" ]; then
  detect_binary
fi
echo "Detected host and ready to start services.  Doing some housekeeping first..."
echo "Using GO_OUT ${GO_OUT}"
export KUBELET_CIDFILE=${TMP_DIR}/kubelet.cid
if [[ "${ENABLE_DAEMON}" = false ]]; then
  trap cleanup EXIT
  trap cleanup INT
fi

KUBECTL=$(kube::util::find-binary "kubectl")

echo "Starting services now!"
if [[ "${START_MODE}" != "kubeletonly" ]]; then
  start_etcd
  set_service_accounts
  start_apiserver
  start_controller_manager
  if [[ "${EXTERNAL_CLOUD_PROVIDER:-}" == "true" ]]; then
    start_cloud_controller_manager
  fi
  start_kubescheduler
  start_dns_addon
  if [[ "${ENABLE_NODELOCAL_DNS:-}" == "true" ]]; then
    start_nodelocaldns
  fi
  start_csi_snapshotter
fi

if [[ "${START_MODE}" != *"nokubelet"* ]]; then
  ## TODO remove this check if/when kubelet is supported on darwin
  # Detect the OS name/arch and display appropriate error.
    case "$(uname -s)" in
      Darwin)
        print_color "kubelet is not currently supported in darwin, kubelet aborted."
        KUBELET_LOG=""
        ;;
      Linux)
        install_cni_if_needed
        start_kubelet
        wait_node_csr
        ;;
      *)
        print_color "Unsupported host OS.  Must be Linux or Mac OS X, kubelet aborted."
        ;;
    esac
fi

if [[ "${START_MODE}" != "kubeletonly" ]]; then
  if [[ "${START_MODE}" != *"nokubeproxy"* ]]; then
    ## TODO remove this check if/when kubelet is supported on darwin
    # Detect the OS name/arch and display appropriate error.
    case "$(uname -s)" in
      Darwin)
        print_color "kubelet is not currently supported in darwin, kube-proxy aborted."
        ;;
      Linux)
        start_kubeproxy
        if [[ "${ENABLE_CLUSTER_DNS}" = true ]]; then
          wait_coredns_available
        fi
        ;;
      *)
        print_color "Unsupported host OS.  Must be Linux or Mac OS X, kube-proxy aborted."
        ;;
    esac
  fi
fi

if [[ "${DEFAULT_STORAGE_CLASS}" = "true" ]]; then
  create_storage_class
fi

print_success

if [[ -n "${DRY_RUN}" ]]; then
  # Ensure that "run" output has been flushed. This waits for anything which might have been started.
  # shellcheck disable=SC2086
  wait ${APISERVER_PID-} ${CTLRMGR_PID-} ${CLOUD_CTLRMGR_PID-} ${KUBELET_PID-} ${PROXY_PID-} ${SCHEDULER_PID-}
  echo "Local etcd is running. Run commands. Press Ctrl-C to shut it down."
  sleep infinity
elif [[ "${ENABLE_DAEMON}" = false ]]; then
  while true; do sleep 1; healthcheck; done
fi

if [[ "${KUBETEST_IN_DOCKER:-}" == "true" ]]; then
  run kubectl "" "${KUBECTL}" config set-cluster local --server=https://localhost:6443 --certificate-authority=/var/run/kubernetes/server-ca.crt
  run kubectl "" "${KUBECTL}" config set-credentials myself --client-key=/var/run/kubernetes/client-admin.key --client-certificate=/var/run/kubernetes/client-admin.crt
  run kubectl "" "${KUBECTL}" config set-context local --cluster=local --user=myself
  run kubectl "" "${KUBECTL}" config use-context local
fi
