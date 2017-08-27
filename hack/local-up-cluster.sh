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

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..

# This command builds and runs a local kubernetes cluster.
# You may need to run this as root to allow kubelet to open docker's socket,
# and to write the test CA in /var/run/kubernetes.
DOCKER_OPTS=${DOCKER_OPTS:-""}
DOCKER=(docker ${DOCKER_OPTS})
DOCKERIZE_KUBELET=${DOCKERIZE_KUBELET:-""}
ALLOW_PRIVILEGED=${ALLOW_PRIVILEGED:-""}
ALLOW_SECURITY_CONTEXT=${ALLOW_SECURITY_CONTEXT:-""}
PSP_ADMISSION=${PSP_ADMISSION:-""}
NODE_ADMISSION=${NODE_ADMISSION:-""}
RUNTIME_CONFIG=${RUNTIME_CONFIG:-""}
KUBELET_AUTHORIZATION_WEBHOOK=${KUBELET_AUTHORIZATION_WEBHOOK:-""}
KUBELET_AUTHENTICATION_WEBHOOK=${KUBELET_AUTHENTICATION_WEBHOOK:-""}
POD_MANIFEST_PATH=${POD_MANIFEST_PATH:-"/var/run/kubernetes/static-pods"}
KUBELET_FLAGS=${KUBELET_FLAGS:-""}
# many dev environments run with swap on, so we don't fail in this env
FAIL_SWAP_ON=${FAIL_SWAP_ON:-"false"}
# Name of the network plugin, eg: "kubenet"
NET_PLUGIN=${NET_PLUGIN:-""}
# Place the config files and binaries required by NET_PLUGIN in these directory,
# eg: "/etc/cni/net.d" for config files, and "/opt/cni/bin" for binaries.
CNI_CONF_DIR=${CNI_CONF_DIR:-""}
CNI_BIN_DIR=${CNI_BIN_DIR:-""}
SERVICE_CLUSTER_IP_RANGE=${SERVICE_CLUSTER_IP_RANGE:-10.0.0.0/24}
FIRST_SERVICE_CLUSTER_IP=${FIRST_SERVICE_CLUSTER_IP:-10.0.0.1}
# if enabled, must set CGROUP_ROOT
CGROUPS_PER_QOS=${CGROUPS_PER_QOS:-true}
# name of the cgroup driver, i.e. cgroupfs or systemd
CGROUP_DRIVER=${CGROUP_DRIVER:-""}
# owner of client certs, default to current user if not specified
USER=${USER:-$(whoami)}

# enables testing eviction scenarios locally.
EVICTION_HARD=${EVICTION_HARD:-"memory.available<100Mi,nodefs.available<10%,nodefs.inodesFree<5%"}
EVICTION_SOFT=${EVICTION_SOFT:-""}
EVICTION_PRESSURE_TRANSITION_PERIOD=${EVICTION_PRESSURE_TRANSITION_PERIOD:-"1m"}

# This script uses docker0 (or whatever container bridge docker is currently using)
# and we don't know the IP of the DNS pod to pass in as --cluster-dns.
# To set this up by hand, set this flag and change DNS_SERVER_IP.
# Note also that you need API_HOST (defined above) for correct DNS.
ENABLE_CLUSTER_DNS=${KUBE_ENABLE_CLUSTER_DNS:-true}
DNS_SERVER_IP=${KUBE_DNS_SERVER_IP:-10.0.0.10}
DNS_DOMAIN=${KUBE_DNS_NAME:-"cluster.local"}
KUBECTL=${KUBECTL:-cluster/kubectl.sh}
WAIT_FOR_URL_API_SERVER=${WAIT_FOR_URL_API_SERVER:-20}
ENABLE_DAEMON=${ENABLE_DAEMON:-false}
HOSTNAME_OVERRIDE=${HOSTNAME_OVERRIDE:-"127.0.0.1"}
CLOUD_PROVIDER=${CLOUD_PROVIDER:-""}
CLOUD_CONFIG=${CLOUD_CONFIG:-""}
FEATURE_GATES=${FEATURE_GATES:-"AllAlpha=false"}
STORAGE_BACKEND=${STORAGE_BACKEND:-"etcd3"}
# enable swagger ui
ENABLE_SWAGGER_UI=${ENABLE_SWAGGER_UI:-false}

# enable kubernetes dashboard
ENABLE_CLUSTER_DASHBOARD=${KUBE_ENABLE_CLUSTER_DASHBOARD:-false}

# enable audit log
ENABLE_APISERVER_BASIC_AUDIT=${ENABLE_APISERVER_BASIC_AUDIT:-false}

# RBAC Mode options
AUTHORIZATION_MODE=${AUTHORIZATION_MODE:-"Node,RBAC"}
KUBECONFIG_TOKEN=${KUBECONFIG_TOKEN:-""}
AUTH_ARGS=${AUTH_ARGS:-""}

# Install a default storage class (enabled by default)
DEFAULT_STORAGE_CLASS=${KUBE_DEFAULT_STORAGE_CLASS:-true}

# start the cache mutation detector by default so that cache mutators will be found
KUBE_CACHE_MUTATION_DETECTOR="${KUBE_CACHE_MUTATION_DETECTOR:-true}"
export KUBE_CACHE_MUTATION_DETECTOR

# panic the server on watch decode errors since they are considered coder mistakes
KUBE_PANIC_WATCH_DECODE_ERROR="${KUBE_PANIC_WATCH_DECODE_ERROR:-true}"
export KUBE_PANIC_WATCH_DECODE_ERROR

ADMISSION_CONTROL=${ADMISSION_CONTROL:-""}
ADMISSION_CONTROL_CONFIG_FILE=${ADMISSION_CONTROL_CONFIG_FILE:-""}

# START_MODE can be 'all', 'kubeletonly', or 'nokubelet'
START_MODE=${START_MODE:-"all"}

# A list of controllers to enable
KUBE_CONTROLLERS="${KUBE_CONTROLLERS:-"*"}"

# sanity check for OpenStack provider
if [ "${CLOUD_PROVIDER}" == "openstack" ]; then
    if [ "${CLOUD_CONFIG}" == "" ]; then
        echo "Missing CLOUD_CONFIG env for OpenStack provider!"
        exit 1
    fi
    if [ ! -f "${CLOUD_CONFIG}" ]; then
        echo "Cloud config ${CLOUD_CONFIG} doesn't exist"
        exit 1
    fi
fi

# warn if users are running with swap allowed
if [ "${FAIL_SWAP_ON}" == "false" ]; then
    echo "WARNING : The kubelet is configured to not fail if swap is enabled; production deployments should disable swap."
fi

if [ "$(id -u)" != "0" ]; then
    echo "WARNING : This script MAY be run as root for docker socket / iptables functionality; if failures occur, retry as root." 2>&1
fi

# Stop right away if the build fails
set -e

source "${KUBE_ROOT}/hack/lib/init.sh"

function usage {
            echo "This script starts a local kube cluster. "
            echo "Example 0: hack/local-up-cluster.sh -h  (this 'help' usage description)"
            echo "Example 1: hack/local-up-cluster.sh -o _output/dockerized/bin/linux/amd64/ (run from docker output)"
            echo "Example 2: hack/local-up-cluster.sh -O (auto-guess the bin path for your platform)"
            echo "Example 3: hack/local-up-cluster.sh (build a local copy of the source)"
}

# This function guesses where the existing cached binary build is for the `-O`
# flag
function guess_built_binary_path {
  local hyperkube_path=$(kube::util::find-binary "hyperkube")
  if [[ -z "${hyperkube_path}" ]]; then
    return
  fi
  echo -n "$(dirname "${hyperkube_path}")"
}

### Allow user to supply the source directory.
GO_OUT=${GO_OUT:-}
while getopts "ho:O" OPTION
do
    case $OPTION in
        o)
            echo "skipping build"
            GO_OUT="$OPTARG"
            echo "using source $GO_OUT"
            ;;
        O)
            GO_OUT=$(guess_built_binary_path)
            if [ "$GO_OUT" == "" ]; then
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

if [ "x$GO_OUT" == "x" ]; then
    make -C "${KUBE_ROOT}" WHAT="cmd/kubectl cmd/hyperkube"
else
    echo "skipped the build."
fi

function test_rkt {
    if [[ -n "${RKT_PATH}" ]]; then
      ${RKT_PATH} list 2> /dev/null 1> /dev/null
      if [ "$?" != "0" ]; then
        echo "Failed to successfully run 'rkt list', please verify that ${RKT_PATH} is the path of rkt binary."
        exit 1
      fi
    else
      rkt list 2> /dev/null 1> /dev/null
      if [ "$?" != "0" ]; then
        echo "Failed to successfully run 'rkt list', please verify that rkt is in \$PATH."
        exit 1
      fi
    fi
}


# Shut down anyway if there's an error.
set +e

API_PORT=${API_PORT:-8080}
API_SECURE_PORT=${API_SECURE_PORT:-6443}

# WARNING: For DNS to work on most setups you should export API_HOST as the docker0 ip address,
API_HOST=${API_HOST:-localhost}
API_HOST_IP=${API_HOST_IP:-"127.0.0.1"}
ADVERTISE_ADDRESS=${ADVERTISE_ADDRESS:-""}
API_BIND_ADDR=${API_BIND_ADDR:-"0.0.0.0"}
EXTERNAL_HOSTNAME=${EXTERNAL_HOSTNAME:-localhost}

KUBELET_HOST=${KUBELET_HOST:-"127.0.0.1"}
# By default only allow CORS for requests on localhost
API_CORS_ALLOWED_ORIGINS=${API_CORS_ALLOWED_ORIGINS:-/127.0.0.1(:[0-9]+)?$,/localhost(:[0-9]+)?$}
KUBELET_PORT=${KUBELET_PORT:-10250}
LOG_LEVEL=${LOG_LEVEL:-3}
# Use to increase verbosity on particular files, e.g. LOG_SPEC=token_controller*=5,other_controller*=4
LOG_SPEC=${LOG_SPEC:-""}
LOG_DIR=${LOG_DIR:-"/tmp"}
CONTAINER_RUNTIME=${CONTAINER_RUNTIME:-"docker"}
CONTAINER_RUNTIME_ENDPOINT=${CONTAINER_RUNTIME_ENDPOINT:-""}
IMAGE_SERVICE_ENDPOINT=${IMAGE_SERVICE_ENDPOINT:-""}
ENABLE_CRI=${ENABLE_CRI:-"true"}
RKT_PATH=${RKT_PATH:-""}
RKT_STAGE1_IMAGE=${RKT_STAGE1_IMAGE:-""}
CHAOS_CHANCE=${CHAOS_CHANCE:-0.0}
CPU_CFS_QUOTA=${CPU_CFS_QUOTA:-true}
ENABLE_HOSTPATH_PROVISIONER=${ENABLE_HOSTPATH_PROVISIONER:-"false"}
CLAIM_BINDER_SYNC_PERIOD=${CLAIM_BINDER_SYNC_PERIOD:-"15s"} # current k8s default
ENABLE_CONTROLLER_ATTACH_DETACH=${ENABLE_CONTROLLER_ATTACH_DETACH:-"true"} # current default
KEEP_TERMINATED_POD_VOLUMES=${KEEP_TERMINATED_POD_VOLUMES:-"true"}
# This is the default dir and filename where the apiserver will generate a self-signed cert
# which should be able to be used as the CA to verify itself
CERT_DIR=${CERT_DIR:-"/var/run/kubernetes"}
ROOT_CA_FILE=${CERT_DIR}/server-ca.crt
ROOT_CA_KEY=${CERT_DIR}/server-ca.key
CLUSTER_SIGNING_CERT_FILE=${CLUSTER_SIGNING_CERT_FILE:-"${ROOT_CA_FILE}"}
CLUSTER_SIGNING_KEY_FILE=${CLUSTER_SIGNING_KEY_FILE:-"${ROOT_CA_KEY}"}

# name of the cgroup driver, i.e. cgroupfs or systemd
if [[ ${CONTAINER_RUNTIME} == "docker" ]]; then
  # default cgroup driver to match what is reported by docker to simplify local development
  if [[ -z ${CGROUP_DRIVER} ]]; then
    # match driver with docker runtime reported value (they must match)
    CGROUP_DRIVER=$(docker info | grep "Cgroup Driver:" | cut -f3- -d' ')
    echo "Kubelet cgroup driver defaulted to use: ${CGROUP_DRIVER}"
  fi
fi



# Ensure CERT_DIR is created for auto-generated crt/key and kubeconfig
mkdir -p "${CERT_DIR}" &>/dev/null || sudo mkdir -p "${CERT_DIR}"
CONTROLPLANE_SUDO=$(test -w "${CERT_DIR}" || echo "sudo -E")

function test_apiserver_off {
    # For the common local scenario, fail fast if server is already running.
    # this can happen if you run local-up-cluster.sh twice and kill etcd in between.
    if [[ "${API_PORT}" -gt "0" ]]; then
        curl --silent -g $API_HOST:$API_PORT
        if [ ! $? -eq 0 ]; then
            echo "API SERVER insecure port is free, proceeding..."
        else
            echo "ERROR starting API SERVER, exiting. Some process on $API_HOST is serving already on $API_PORT"
            exit 1
        fi
    fi

    curl --silent -k -g $API_HOST:$API_SECURE_PORT
    if [ ! $? -eq 0 ]; then
        echo "API SERVER secure port is free, proceeding..."
    else
        echo "ERROR starting API SERVER, exiting. Some process on $API_HOST is serving already on $API_SECURE_PORT"
        exit 1
    fi
}

function detect_binary {
    # Detect the OS name/arch so that we can find our binary
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

   GO_OUT="${KUBE_ROOT}/_output/local/bin/${host_os}/${host_arch}"
}

cleanup_dockerized_kubelet()
{
  if [[ -e $KUBELET_CIDFILE ]]; then
    docker kill $(<$KUBELET_CIDFILE) > /dev/null
    rm -f $KUBELET_CIDFILE
  fi
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
  [[ -n "${APISERVER_PID-}" ]] && APISERVER_PIDS=$(pgrep -P ${APISERVER_PID} ; ps -o pid= -p ${APISERVER_PID})
  [[ -n "${APISERVER_PIDS-}" ]] && sudo kill ${APISERVER_PIDS}

  # Check if the controller-manager is still running
  [[ -n "${CTLRMGR_PID-}" ]] && CTLRMGR_PIDS=$(pgrep -P ${CTLRMGR_PID} ; ps -o pid= -p ${CTLRMGR_PID})
  [[ -n "${CTLRMGR_PIDS-}" ]] && sudo kill ${CTLRMGR_PIDS}

  if [[ -n "$DOCKERIZE_KUBELET" ]]; then
    cleanup_dockerized_kubelet
  else
    # Check if the kubelet is still running
    [[ -n "${KUBELET_PID-}" ]] && KUBELET_PIDS=$(pgrep -P ${KUBELET_PID} ; ps -o pid= -p ${KUBELET_PID})
    [[ -n "${KUBELET_PIDS-}" ]] && sudo kill ${KUBELET_PIDS}
  fi

  # Check if the proxy is still running
  [[ -n "${PROXY_PID-}" ]] && PROXY_PIDS=$(pgrep -P ${PROXY_PID} ; ps -o pid= -p ${PROXY_PID})
  [[ -n "${PROXY_PIDS-}" ]] && sudo kill ${PROXY_PIDS}

  # Check if the scheduler is still running
  [[ -n "${SCHEDULER_PID-}" ]] && SCHEDULER_PIDS=$(pgrep -P ${SCHEDULER_PID} ; ps -o pid= -p ${SCHEDULER_PID})
  [[ -n "${SCHEDULER_PIDS-}" ]] && sudo kill ${SCHEDULER_PIDS}

  # Check if the etcd is still running
  [[ -n "${ETCD_PID-}" ]] && kube::etcd::stop
  [[ -n "${ETCD_DIR-}" ]] && kube::etcd::clean_etcd_dir

  exit 0
}

function warning {
  message=$1

  echo $(tput bold)$(tput setaf 1)
  echo "WARNING: ${message}"
  echo $(tput sgr0)
}

function start_etcd {
    echo "Starting etcd"
    kube::etcd::start
}

function set_service_accounts {
    SERVICE_ACCOUNT_LOOKUP=${SERVICE_ACCOUNT_LOOKUP:-true}
    SERVICE_ACCOUNT_KEY=${SERVICE_ACCOUNT_KEY:-/tmp/kube-serviceaccount.key}
    # Generate ServiceAccount key if needed
    if [[ ! -f "${SERVICE_ACCOUNT_KEY}" ]]; then
      mkdir -p "$(dirname ${SERVICE_ACCOUNT_KEY})"
      openssl genrsa -out "${SERVICE_ACCOUNT_KEY}" 2048 2>/dev/null
    fi
}

function start_apiserver {
    security_admission=""
    if [[ -z "${ALLOW_SECURITY_CONTEXT}" ]]; then
      security_admission=",SecurityContextDeny"
    fi
    if [[ -n "${PSP_ADMISSION}" ]]; then
      security_admission=",PodSecurityPolicy"
    fi
    if [[ -n "${NODE_ADMISSION}" ]]; then
      security_admission=",NodeRestriction"
    fi

    # Admission Controllers to invoke prior to persisting objects in cluster
    ADMISSION_CONTROL=Initializers,NamespaceLifecycle,LimitRanger,ServiceAccount${security_admission},DefaultStorageClass,DefaultTolerationSeconds,ResourceQuota
    # This is the default dir and filename where the apiserver will generate a self-signed cert
    # which should be able to be used as the CA to verify itself

    audit_arg=""
    APISERVER_BASIC_AUDIT_LOG=""
    if [[ "${ENABLE_APISERVER_BASIC_AUDIT:-}" = true ]]; then
        # We currently only support enabling with a fixed path and with built-in log
        # rotation "disabled" (large value) so it behaves like kube-apiserver.log.
        # External log rotation should be set up the same as for kube-apiserver.log.
        APISERVER_BASIC_AUDIT_LOG=/tmp/kube-apiserver-audit.log
        audit_arg=" --audit-log-path=${APISERVER_BASIC_AUDIT_LOG}"
        audit_arg+=" --audit-log-maxage=0"
        audit_arg+=" --audit-log-maxbackup=0"
        # Lumberjack doesn't offer any way to disable size-based rotation. It also
        # has an in-memory counter that doesn't notice if you truncate the file.
        # 2000000000 (in MiB) is a large number that fits in 31 bits. If the log
        # grows at 10MiB/s (~30K QPS), it will rotate after ~6 years if apiserver
        # never restarts. Please manually restart apiserver before this time.
        audit_arg+=" --audit-log-maxsize=2000000000"
    fi

    swagger_arg=""
    if [[ "${ENABLE_SWAGGER_UI}" = true ]]; then
      swagger_arg="--enable-swagger-ui=true "
    fi

    authorizer_arg=""
    if [[ -n "${AUTHORIZATION_MODE}" ]]; then
      authorizer_arg="--authorization-mode=${AUTHORIZATION_MODE} "
    fi
    priv_arg=""
    if [[ -n "${ALLOW_PRIVILEGED}" ]]; then
      priv_arg="--allow-privileged "
    fi

    if [[ ${ADMISSION_CONTROL} == *"Initializers"* ]]; then
        if [[ -n "${RUNTIME_CONFIG}" ]]; then
          RUNTIME_CONFIG+=","
        fi
        RUNTIME_CONFIG+="admissionregistration.k8s.io/v1alpha1"
    fi

    runtime_config=""
    if [[ -n "${RUNTIME_CONFIG}" ]]; then
      runtime_config="--runtime-config=${RUNTIME_CONFIG}"
    fi

    # Let the API server pick a default address when API_HOST_IP
    # is set to 127.0.0.1
    advertise_address=""
    if [[ "${API_HOST_IP}" != "127.0.0.1" ]]; then
        advertise_address="--advertise_address=${API_HOST_IP}"
    fi
    if [[ "${ADVERTISE_ADDRESS}" != "" ]] ; then
        advertise_address="--advertise_address=${ADVERTISE_ADDRESS}"
    fi

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
    kube::util::create_serving_certkey "${CONTROLPLANE_SUDO}" "${CERT_DIR}" "server-ca" kube-apiserver kubernetes.default kubernetes.default.svc "localhost" ${API_HOST_IP} ${API_HOST} ${FIRST_SERVICE_CLUSTER_IP}

    # Create client certs signed with client-ca, given id, given CN and a number of groups
    kube::util::create_client_certkey "${CONTROLPLANE_SUDO}" "${CERT_DIR}" 'client-ca' kubelet system:node:${HOSTNAME_OVERRIDE} system:nodes
    kube::util::create_client_certkey "${CONTROLPLANE_SUDO}" "${CERT_DIR}" 'client-ca' kube-proxy system:kube-proxy system:nodes
    kube::util::create_client_certkey "${CONTROLPLANE_SUDO}" "${CERT_DIR}" 'client-ca' controller system:kube-controller-manager
    kube::util::create_client_certkey "${CONTROLPLANE_SUDO}" "${CERT_DIR}" 'client-ca' scheduler  system:kube-scheduler
    kube::util::create_client_certkey "${CONTROLPLANE_SUDO}" "${CERT_DIR}" 'client-ca' admin system:admin system:masters

    # Create matching certificates for kube-aggregator
    kube::util::create_serving_certkey "${CONTROLPLANE_SUDO}" "${CERT_DIR}" "server-ca" kube-aggregator api.kube-public.svc "localhost" ${API_HOST_IP}
    kube::util::create_client_certkey "${CONTROLPLANE_SUDO}" "${CERT_DIR}" request-header-ca auth-proxy system:auth-proxy
    # TODO remove masters and add rolebinding
    kube::util::create_client_certkey "${CONTROLPLANE_SUDO}" "${CERT_DIR}" 'client-ca' kube-aggregator system:kube-aggregator system:masters
    kube::util::write_client_kubeconfig "${CONTROLPLANE_SUDO}" "${CERT_DIR}" "${ROOT_CA_FILE}" "${API_HOST}" "${API_SECURE_PORT}" kube-aggregator


    APISERVER_LOG=${LOG_DIR}/kube-apiserver.log
    ${CONTROLPLANE_SUDO} "${GO_OUT}/hyperkube" apiserver ${swagger_arg} ${audit_arg} ${authorizer_arg} ${priv_arg} ${runtime_config}\
      ${advertise_address} \
      --v=${LOG_LEVEL} \
      --vmodule="${LOG_SPEC}" \
      --cert-dir="${CERT_DIR}" \
      --client-ca-file="${CERT_DIR}/client-ca.crt" \
      --service-account-key-file="${SERVICE_ACCOUNT_KEY}" \
      --service-account-lookup="${SERVICE_ACCOUNT_LOOKUP}" \
      --admission-control="${ADMISSION_CONTROL}" \
      --admission-control-config-file="${ADMISSION_CONTROL_CONFIG_FILE}" \
      --bind-address="${API_BIND_ADDR}" \
      --secure-port="${API_SECURE_PORT}" \
      --tls-cert-file="${CERT_DIR}/serving-kube-apiserver.crt" \
      --tls-private-key-file="${CERT_DIR}/serving-kube-apiserver.key" \
      --tls-ca-file="${CERT_DIR}/server-ca.crt" \
      --insecure-bind-address="${API_HOST_IP}" \
      --insecure-port="${API_PORT}" \
      --storage-backend=${STORAGE_BACKEND} \
      --etcd-servers="http://${ETCD_HOST}:${ETCD_PORT}" \
      --service-cluster-ip-range="${SERVICE_CLUSTER_IP_RANGE}" \
      --feature-gates="${FEATURE_GATES}" \
      --external-hostname="${EXTERNAL_HOSTNAME}" \
      --cloud-provider="${CLOUD_PROVIDER}" \
      --cloud-config="${CLOUD_CONFIG}" \
      --requestheader-username-headers=X-Remote-User \
      --requestheader-group-headers=X-Remote-Group \
      --requestheader-extra-headers-prefix=X-Remote-Extra- \
      --requestheader-client-ca-file="${CERT_DIR}/request-header-ca.crt" \
      --requestheader-allowed-names=system:auth-proxy \
      --proxy-client-cert-file="${CERT_DIR}/client-auth-proxy.crt" \
      --proxy-client-key-file="${CERT_DIR}/client-auth-proxy.key" \
      --cors-allowed-origins="${API_CORS_ALLOWED_ORIGINS}" >"${APISERVER_LOG}" 2>&1 &
    APISERVER_PID=$!

    # Wait for kube-apiserver to come up before launching the rest of the components.
    echo "Waiting for apiserver to come up"
    # this uses the API port because if you don't have any authenticator, you can't seem to use the secure port at all.
    # this matches what happened with the combination in 1.4.
    # TODO change this conditionally based on whether API_PORT is on or off
    kube::util::wait_for_url "https://${API_HOST_IP}:${API_SECURE_PORT}/healthz" "apiserver: " 1 ${WAIT_FOR_URL_API_SERVER} \
        || { echo "check apiserver logs: ${APISERVER_LOG}" ; exit 1 ; }

    # Create kubeconfigs for all components, using client certs
    kube::util::write_client_kubeconfig "${CONTROLPLANE_SUDO}" "${CERT_DIR}" "${ROOT_CA_FILE}" "${API_HOST}" "${API_SECURE_PORT}" admin
    ${CONTROLPLANE_SUDO} chown "${USER}" "${CERT_DIR}/client-admin.key" # make readable for kubectl
    kube::util::write_client_kubeconfig "${CONTROLPLANE_SUDO}" "${CERT_DIR}" "${ROOT_CA_FILE}" "${API_HOST}" "${API_SECURE_PORT}" kubelet
    kube::util::write_client_kubeconfig "${CONTROLPLANE_SUDO}" "${CERT_DIR}" "${ROOT_CA_FILE}" "${API_HOST}" "${API_SECURE_PORT}" kube-proxy
    kube::util::write_client_kubeconfig "${CONTROLPLANE_SUDO}" "${CERT_DIR}" "${ROOT_CA_FILE}" "${API_HOST}" "${API_SECURE_PORT}" controller
    kube::util::write_client_kubeconfig "${CONTROLPLANE_SUDO}" "${CERT_DIR}" "${ROOT_CA_FILE}" "${API_HOST}" "${API_SECURE_PORT}" scheduler

    if [[ -z "${AUTH_ARGS}" ]]; then
        AUTH_ARGS="--client-key=${CERT_DIR}/client-admin.key --client-certificate=${CERT_DIR}/client-admin.crt"
    fi

    ${CONTROLPLANE_SUDO} cp "${CERT_DIR}/admin.kubeconfig" "${CERT_DIR}/admin-kube-aggregator.kubeconfig"
    ${CONTROLPLANE_SUDO} chown $(whoami) "${CERT_DIR}/admin-kube-aggregator.kubeconfig"
    ${KUBECTL} config set-cluster local-up-cluster --kubeconfig="${CERT_DIR}/admin-kube-aggregator.kubeconfig" --server="https://${API_HOST_IP}:31090"
    echo "use 'kubectl --kubeconfig=${CERT_DIR}/admin-kube-aggregator.kubeconfig' to use the aggregated API server"

}

function start_controller_manager {
    node_cidr_args=""
    if [[ "${NET_PLUGIN}" == "kubenet" ]]; then
      node_cidr_args="--allocate-node-cidrs=true --cluster-cidr=10.1.0.0/16 "
    fi

    CTLRMGR_LOG=${LOG_DIR}/kube-controller-manager.log
    ${CONTROLPLANE_SUDO} "${GO_OUT}/hyperkube" controller-manager \
      --v=${LOG_LEVEL} \
      --vmodule="${LOG_SPEC}" \
      --service-account-private-key-file="${SERVICE_ACCOUNT_KEY}" \
      --root-ca-file="${ROOT_CA_FILE}" \
      --cluster-signing-cert-file="${CLUSTER_SIGNING_CERT_FILE}" \
      --cluster-signing-key-file="${CLUSTER_SIGNING_KEY_FILE}" \
      --enable-hostpath-provisioner="${ENABLE_HOSTPATH_PROVISIONER}" \
      ${node_cidr_args} \
      --pvclaimbinder-sync-period="${CLAIM_BINDER_SYNC_PERIOD}" \
      --feature-gates="${FEATURE_GATES}" \
      --cloud-provider="${CLOUD_PROVIDER}" \
      --cloud-config="${CLOUD_CONFIG}" \
      --kubeconfig "$CERT_DIR"/controller.kubeconfig \
      --use-service-account-credentials \
      --controllers="${KUBE_CONTROLLERS}" \
      --master="https://${API_HOST}:${API_SECURE_PORT}" >"${CTLRMGR_LOG}" 2>&1 &
    CTLRMGR_PID=$!
}

function start_kubelet {
    KUBELET_LOG=${LOG_DIR}/kubelet.log
    mkdir -p "${POD_MANIFEST_PATH}" &>/dev/null || sudo mkdir -p "${POD_MANIFEST_PATH}"

    priv_arg=""
    if [[ -n "${ALLOW_PRIVILEGED}" ]]; then
      priv_arg="--allow-privileged "
    fi

    mkdir -p "/var/lib/kubelet" &>/dev/null || sudo mkdir -p "/var/lib/kubelet"
    if [[ -z "${DOCKERIZE_KUBELET}" ]]; then
      # Enable dns
      if [[ "${ENABLE_CLUSTER_DNS}" = true ]]; then
         dns_args="--cluster-dns=${DNS_SERVER_IP} --cluster-domain=${DNS_DOMAIN}"
      else
         # To start a private DNS server set ENABLE_CLUSTER_DNS and
         # DNS_SERVER_IP/DOMAIN. This will at least provide a working
         # DNS server for real world hostnames.
         dns_args="--cluster-dns=8.8.8.8"
      fi

      net_plugin_args=""
      if [[ -n "${NET_PLUGIN}" ]]; then
        net_plugin_args="--network-plugin=${NET_PLUGIN}"
      fi

      auth_args=""
      if [[ -n "${KUBELET_AUTHORIZATION_WEBHOOK:-}" ]]; then
        auth_args="${auth_args} --authorization-mode=Webhook"
      fi
      if [[ -n "${KUBELET_AUTHENTICATION_WEBHOOK:-}" ]]; then
        auth_args="${auth_args} --authentication-token-webhook"
      fi
      if [[ -n "${CLIENT_CA_FILE:-}" ]]; then
        auth_args="${auth_args} --client-ca-file=${CLIENT_CA_FILE}"
      fi

      cni_conf_dir_args=""
      if [[ -n "${CNI_CONF_DIR}" ]]; then
        cni_conf_dir_args="--cni-conf-dir=${CNI_CONF_DIR}"
      fi

      cni_bin_dir_args=""
      if [[ -n "${CNI_BIN_DIR}" ]]; then
        cni_bin_dir_args="--cni-bin-dir=${CNI_BIN_DIR}"
      fi

      container_runtime_endpoint_args=""
      if [[ -n "${CONTAINER_RUNTIME_ENDPOINT}" ]]; then
        container_runtime_endpoint_args="--container-runtime-endpoint=${CONTAINER_RUNTIME_ENDPOINT}"
      fi

      image_service_endpoint_args=""
      if [[ -n "${IMAGE_SERVICE_ENDPOINT}" ]]; then
        image_service_endpoint_args="--image-service-endpoint=${IMAGE_SERVICE_ENDPOINT}"
      fi

      sudo -E "${GO_OUT}/hyperkube" kubelet ${priv_arg}\
        --v=${LOG_LEVEL} \
        --vmodule="${LOG_SPEC}" \
        --chaos-chance="${CHAOS_CHANCE}" \
        --container-runtime="${CONTAINER_RUNTIME}" \
        --rkt-path="${RKT_PATH}" \
        --rkt-stage1-image="${RKT_STAGE1_IMAGE}" \
        --hostname-override="${HOSTNAME_OVERRIDE}" \
        --cloud-provider="${CLOUD_PROVIDER}" \
        --cloud-config="${CLOUD_CONFIG}" \
        --address="${KUBELET_HOST}" \
        --kubeconfig "$CERT_DIR"/kubelet.kubeconfig \
        --feature-gates="${FEATURE_GATES}" \
        --cpu-cfs-quota=${CPU_CFS_QUOTA} \
        --enable-controller-attach-detach="${ENABLE_CONTROLLER_ATTACH_DETACH}" \
        --cgroups-per-qos=${CGROUPS_PER_QOS} \
        --cgroup-driver=${CGROUP_DRIVER} \
        --keep-terminated-pod-volumes=${KEEP_TERMINATED_POD_VOLUMES} \
        --eviction-hard=${EVICTION_HARD} \
        --eviction-soft=${EVICTION_SOFT} \
        --eviction-pressure-transition-period=${EVICTION_PRESSURE_TRANSITION_PERIOD} \
        --pod-manifest-path="${POD_MANIFEST_PATH}" \
        --fail-swap-on="${FAIL_SWAP_ON}" \
        ${auth_args} \
        ${dns_args} \
        ${cni_conf_dir_args} \
        ${cni_bin_dir_args} \
        ${net_plugin_args} \
        ${container_runtime_endpoint_args} \
        ${image_service_endpoint_args} \
        --port="$KUBELET_PORT" \
	${KUBELET_FLAGS} >"${KUBELET_LOG}" 2>&1 &
      KUBELET_PID=$!
      # Quick check that kubelet is running.
      if ps -p $KUBELET_PID > /dev/null ; then
	echo "kubelet ( $KUBELET_PID ) is running."
      else
	cat ${KUBELET_LOG} ; exit 1
      fi
    else
      # Docker won't run a container with a cidfile (container id file)
      # unless that file does not already exist; clean up an existing
      # dockerized kubelet that might be running.
      cleanup_dockerized_kubelet
      cred_bind=""
      # path to cloud credentials.
      cloud_cred=""
      if [ "${CLOUD_PROVIDER}" == "aws" ]; then
          cloud_cred="${HOME}/.aws/credentials"
      fi
      if [ "${CLOUD_PROVIDER}" == "gce" ]; then
          cloud_cred="${HOME}/.config/gcloud"
      fi
      if [ "${CLOUD_PROVIDER}" == "openstack" ]; then
          cloud_cred="${CLOUD_CONFIG}"
      fi
      if  [[ -n "${cloud_cred}" ]]; then
          cred_bind="--volume=${cloud_cred}:${cloud_cred}:ro"
      fi

      docker run \
        --volume=/:/rootfs:ro \
        --volume=/var/run:/var/run:rw \
        --volume=/sys:/sys:ro \
        --volume=/var/lib/docker/:/var/lib/docker:ro \
        --volume=/var/lib/kubelet/:/var/lib/kubelet:rw \
        --volume=/dev:/dev \
        --volume=/run/xtables.lock:/run/xtables.lock:rw \
        ${cred_bind} \
        --net=host \
        --privileged=true \
        -i \
        --cidfile=$KUBELET_CIDFILE \
        gcr.io/google_containers/kubelet \
        /kubelet --v=${LOG_LEVEL} --containerized ${priv_arg}--chaos-chance="${CHAOS_CHANCE}" --pod-manifest-path="${POD_MANIFEST_PATH}" --hostname-override="${HOSTNAME_OVERRIDE}" --cloud-provider="${CLOUD_PROVIDER}" --cloud-config="${CLOUD_CONFIG}" \ --address="127.0.0.1" --kubeconfig "$CERT_DIR"/kubelet.kubeconfig --port="$KUBELET_PORT"  --enable-controller-attach-detach="${ENABLE_CONTROLLER_ATTACH_DETACH}" &> $KUBELET_LOG &
    fi
}

function start_kubeproxy {
    PROXY_LOG=${LOG_DIR}/kube-proxy.log

    cat <<EOF > /tmp/kube-proxy.yaml
apiVersion: componentconfig/v1alpha1
kind: KubeProxyConfiguration
clientConnection:
  kubeconfig: ${CERT_DIR}/kube-proxy.kubeconfig
hostnameOverride: ${HOSTNAME_OVERRIDE}
featureGates: ${FEATURE_GATES}
EOF

    sudo "${GO_OUT}/hyperkube" proxy \
      --config=/tmp/kube-proxy.yaml \
      --master="https://${API_HOST}:${API_SECURE_PORT}" >"${PROXY_LOG}" \
      --v=${LOG_LEVEL} 2>&1 &
    PROXY_PID=$!

    SCHEDULER_LOG=${LOG_DIR}/kube-scheduler.log
    ${CONTROLPLANE_SUDO} "${GO_OUT}/hyperkube" scheduler \
      --v=${LOG_LEVEL} \
      --kubeconfig "$CERT_DIR"/scheduler.kubeconfig \
      --master="https://${API_HOST}:${API_SECURE_PORT}" >"${SCHEDULER_LOG}" 2>&1 &
    SCHEDULER_PID=$!
}

function start_kubedns {
    if [[ "${ENABLE_CLUSTER_DNS}" = true ]]; then
        echo "Creating kube-system namespace"
        sed -e "s/{{ pillar\['dns_domain'\] }}/${DNS_DOMAIN}/g" "${KUBE_ROOT}/cluster/addons/dns/kubedns-controller.yaml.in" >| kubedns-deployment.yaml
        sed -e "s/{{ pillar\['dns_server'\] }}/${DNS_SERVER_IP}/g" "${KUBE_ROOT}/cluster/addons/dns/kubedns-svc.yaml.in" >| kubedns-svc.yaml

        # TODO update to dns role once we have one.
        ${KUBECTL} --kubeconfig="${CERT_DIR}/admin.kubeconfig" create clusterrolebinding system:kube-dns --clusterrole=cluster-admin --serviceaccount=kube-system:default
        # use kubectl to create kubedns deployment and service
        ${KUBECTL} --kubeconfig="${CERT_DIR}/admin.kubeconfig" --namespace=kube-system create -f ${KUBE_ROOT}/cluster/addons/dns/kubedns-sa.yaml
        ${KUBECTL} --kubeconfig="${CERT_DIR}/admin.kubeconfig" --namespace=kube-system create -f ${KUBE_ROOT}/cluster/addons/dns/kubedns-cm.yaml
        ${KUBECTL} --kubeconfig="${CERT_DIR}/admin.kubeconfig" --namespace=kube-system create -f kubedns-deployment.yaml
        ${KUBECTL} --kubeconfig="${CERT_DIR}/admin.kubeconfig" --namespace=kube-system create -f kubedns-svc.yaml
        echo "Kube-dns deployment and service successfully deployed."
        rm  kubedns-deployment.yaml kubedns-svc.yaml
    fi
}

function start_kubedashboard {
    if [[ "${ENABLE_CLUSTER_DASHBOARD}" = true ]]; then
        echo "Creating kubernetes-dashboard"
        # use kubectl to create the dashboard
        ${KUBECTL} --kubeconfig="${CERT_DIR}/admin.kubeconfig" create -f ${KUBE_ROOT}/cluster/addons/dashboard/dashboard-controller.yaml
        ${KUBECTL} --kubeconfig="${CERT_DIR}/admin.kubeconfig" create -f ${KUBE_ROOT}/cluster/addons/dashboard/dashboard-service.yaml
        echo "kubernetes-dashboard deployment and service successfully deployed."
    fi
}

function create_psp_policy {
    echo "Create podsecuritypolicy policies for RBAC."
    ${KUBECTL} --kubeconfig="${CERT_DIR}/admin.kubeconfig" create -f ${KUBE_ROOT}/examples/podsecuritypolicy/rbac/policies.yaml
    ${KUBECTL} --kubeconfig="${CERT_DIR}/admin.kubeconfig" create -f ${KUBE_ROOT}/examples/podsecuritypolicy/rbac/roles.yaml
    ${KUBECTL} --kubeconfig="${CERT_DIR}/admin.kubeconfig" create -f ${KUBE_ROOT}/examples/podsecuritypolicy/rbac/bindings.yaml
}

function create_storage_class {
    if [ -z "$CLOUD_PROVIDER" ]; then
        CLASS_FILE=${KUBE_ROOT}/cluster/addons/storage-class/local/default.yaml
    else
        CLASS_FILE=${KUBE_ROOT}/cluster/addons/storage-class/${CLOUD_PROVIDER}/default.yaml
    fi

    if [ -e $CLASS_FILE ]; then
        echo "Create default storage class for $CLOUD_PROVIDER"
        ${KUBECTL} --kubeconfig="${CERT_DIR}/admin.kubeconfig" create -f $CLASS_FILE
    else
        echo "No storage class available for $CLOUD_PROVIDER."
    fi
}

function print_success {
if [[ "${START_MODE}" != "kubeletonly" ]]; then
  cat <<EOF
Local Kubernetes cluster is running. Press Ctrl-C to shut it down.

Logs:
  ${APISERVER_LOG:-}
  ${CTLRMGR_LOG:-}
  ${PROXY_LOG:-}
  ${SCHEDULER_LOG:-}
EOF
fi

if [[ "${ENABLE_APISERVER_BASIC_AUDIT:-}" = true ]]; then
  echo "  ${APISERVER_BASIC_AUDIT_LOG}"
fi

if [[ "${START_MODE}" == "all" ]]; then
  echo "  ${KUBELET_LOG}"
elif [[ "${START_MODE}" == "nokubelet" ]]; then
  echo
  echo "No kubelet was started because you set START_MODE=nokubelet"
  echo "Run this script again with START_MODE=kubeletonly to run a kubelet"
fi

if [[ "${START_MODE}" != "kubeletonly" ]]; then
  echo
  cat <<EOF
To start using your cluster, you can open up another terminal/tab and run:

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

# validate that etcd is: not running, in path, and has minimum required version.
if [[ "${START_MODE}" != "kubeletonly" ]]; then
  kube::etcd::validate
fi

if [ "${CONTAINER_RUNTIME}" == "docker" ] && ! kube::util::ensure_docker_daemon_connectivity; then
  exit 1
fi

if [[ "${CONTAINER_RUNTIME}" == "rkt" ]]; then
  test_rkt
fi

if [[ "${START_MODE}" != "kubeletonly" ]]; then
  test_apiserver_off
fi

kube::util::test_openssl_installed
kube::util::ensure-cfssl

### IF the user didn't supply an output/ for the build... Then we detect.
if [ "$GO_OUT" == "" ]; then
  detect_binary
fi
echo "Detected host and ready to start services.  Doing some housekeeping first..."
echo "Using GO_OUT $GO_OUT"
KUBELET_CIDFILE=/tmp/kubelet.cid
if [[ "${ENABLE_DAEMON}" = false ]]; then
  trap cleanup EXIT
fi

echo "Starting services now!"
if [[ "${START_MODE}" != "kubeletonly" ]]; then
  start_etcd
  set_service_accounts
  start_apiserver
  start_controller_manager
  start_kubeproxy
  start_kubedns
  start_kubedashboard
fi

if [[ "${START_MODE}" != "nokubelet" ]]; then
  ## TODO remove this check if/when kubelet is supported on darwin
  # Detect the OS name/arch and display appropriate error.
    case "$(uname -s)" in
      Darwin)
        warning "kubelet is not currently supported in darwin, kubelet aborted."
        KUBELET_LOG=""
        ;;
      Linux)
        start_kubelet
        ;;
      *)
        warning "Unsupported host OS.  Must be Linux or Mac OS X, kubelet aborted."
        ;;
    esac
fi

if [[ -n "${PSP_ADMISSION}" && "${AUTHORIZATION_MODE}" = *RBAC* ]]; then
  create_psp_policy
fi

if [[ "$DEFAULT_STORAGE_CLASS" = "true" ]]; then
  create_storage_class
fi

print_success

if [[ "${ENABLE_DAEMON}" = false ]]; then
  while true; do sleep 1; done
fi
