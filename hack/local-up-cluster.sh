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
RUNTIME_CONFIG=${RUNTIME_CONFIG:-""}
KUBELET_AUTHORIZATION_WEBHOOK=${KUBELET_AUTHORIZATION_WEBHOOK:-""}
KUBELET_AUTHENTICATION_WEBHOOK=${KUBELET_AUTHENTICATION_WEBHOOK:-""}
# Name of the network plugin, eg: "kubenet"
NET_PLUGIN=${NET_PLUGIN:-""}
# Place the binaries required by NET_PLUGIN in this directory, eg: "/home/kubernetes/bin".
NET_PLUGIN_DIR=${NET_PLUGIN_DIR:-""}
SERVICE_CLUSTER_IP_RANGE=${SERVICE_CLUSTER_IP_RANGE:-10.0.0.0/24}
# if enabled, must set CGROUP_ROOT
EXPERIMENTAL_CGROUPS_PER_QOS=${EXPERIMENTAL_CGROUPS_PER_QOS:-false}
# this is not defaulted to preserve backward compatibility.
# if EXPERIMENTAL_CGROUPS_PER_QOS is enabled, recommend setting to /
CGROUP_ROOT=${CGROUP_ROOT:-""}
# name of the cgroup driver, i.e. cgroupfs or systemd
CGROUP_DRIVER=${CGROUP_DRIVER:-""}

# We disable cluster DNS by default because this script uses docker0 (or whatever
# container bridge docker is currently using) and we don't know the IP of the
# DNS pod to pass in as --cluster-dns. To set this up by hand, set this flag
# and change DNS_SERVER_IP to the appropriate IP.
ENABLE_CLUSTER_DNS=${KUBE_ENABLE_CLUSTER_DNS:-false}
DNS_SERVER_IP=${KUBE_DNS_SERVER_IP:-10.0.0.10}
DNS_DOMAIN=${KUBE_DNS_NAME:-"cluster.local"}
KUBECTL=${KUBECTL:-cluster/kubectl.sh}
WAIT_FOR_URL_API_SERVER=${WAIT_FOR_URL_API_SERVER:-10}
ENABLE_DAEMON=${ENABLE_DAEMON:-false}
HOSTNAME_OVERRIDE=${HOSTNAME_OVERRIDE:-"127.0.0.1"}
CLOUD_PROVIDER=${CLOUD_PROVIDER:-""}
CLOUD_CONFIG=${CLOUD_CONFIG:-""}
FEATURE_GATES=${FEATURE_GATES:-"AllAlpha=true"}

# RBAC Mode options
ALLOW_ANY_TOKEN=${ALLOW_ANY_TOKEN:-false}
ENABLE_RBAC=${ENABLE_RBAC:-false}
KUBECONFIG_TOKEN=${KUBECONFIG_TOKEN:-""}
AUTH_ARGS=${AUTH_ARGS:-""}

# start the cache mutation detector by default so that cache mutators will be found
KUBE_CACHE_MUTATION_DETECTOR="${KUBE_CACHE_MUTATION_DETECTOR:-true}"
export KUBE_CACHE_MUTATION_DETECTOR


# START_MODE can be 'all', 'kubeletonly', or 'nokubelet'
START_MODE=${START_MODE:-"all"}

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
            if [ $GO_OUT == "" ]; then
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
    make -C "${KUBE_ROOT}" WHAT="cmd/kubectl cmd/hyperkube cmd/kube-aggregator"
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
API_HOST=${API_HOST:-localhost}
API_HOST_IP=${API_HOST_IP:-"127.0.0.1"}
API_BIND_ADDR=${API_BIND_ADDR:-"0.0.0.0"}
KUBELET_HOST=${KUBELET_HOST:-"127.0.0.1"}
# By default only allow CORS for requests on localhost
API_CORS_ALLOWED_ORIGINS=${API_CORS_ALLOWED_ORIGINS:-/127.0.0.1(:[0-9]+)?$,/localhost(:[0-9]+)?$}
KUBELET_PORT=${KUBELET_PORT:-10250}
LOG_LEVEL=${LOG_LEVEL:-3}
CONTAINER_RUNTIME=${CONTAINER_RUNTIME:-"docker"}
CONTAINER_RUNTIME_ENDPOINT=${CONTAINER_RUNTIME_ENDPOINT:-""}
IMAGE_SERVICE_ENDPOINT=${IMAGE_SERVICE_ENDPOINT:-""}
RKT_PATH=${RKT_PATH:-""}
RKT_STAGE1_IMAGE=${RKT_STAGE1_IMAGE:-""}
CHAOS_CHANCE=${CHAOS_CHANCE:-0.0}
CPU_CFS_QUOTA=${CPU_CFS_QUOTA:-true}
ENABLE_HOSTPATH_PROVISIONER=${ENABLE_HOSTPATH_PROVISIONER:-"false"}
CLAIM_BINDER_SYNC_PERIOD=${CLAIM_BINDER_SYNC_PERIOD:-"15s"} # current k8s default
ENABLE_CONTROLLER_ATTACH_DETACH=${ENABLE_CONTROLLER_ATTACH_DETACH:-"true"} # current default
# This is the default dir and filename where the apiserver will generate a self-signed cert
# which should be able to be used as the CA to verify itself
CERT_DIR=${CERT_DIR:-"/var/run/kubernetes"}
ROOT_CA_FILE=$CERT_DIR/apiserver.crt
EXPERIMENTAL_CRI=${EXPERIMENTAL_CRI:-"false"}


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
  # if [[ "${ENABLE_CLUSTER_DNS}" = true ]]; then
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

function start_etcd {
    echo "Starting etcd"
    kube::etcd::start
}

function set_service_accounts {
    SERVICE_ACCOUNT_LOOKUP=${SERVICE_ACCOUNT_LOOKUP:-false}
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

    # Admission Controllers to invoke prior to persisting objects in cluster
    ADMISSION_CONTROL=NamespaceLifecycle,LimitRanger,ServiceAccount${security_admission},ResourceQuota,DefaultStorageClass

    # This is the default dir and filename where the apiserver will generate a self-signed cert
    # which should be able to be used as the CA to verify itself

    anytoken_arg=""
    if [[ "${ALLOW_ANY_TOKEN}" = true ]]; then
      anytoken_arg="--insecure-allow-any-token "
      KUBECONFIG_TOKEN="${KUBECONFIG_TOKEN:-system:admin/system:masters}"
    fi
    authorizer_arg=""
    if [[ "${ENABLE_RBAC}" = true ]]; then
      authorizer_arg="--authorization-mode=RBAC "
    fi
    priv_arg=""
    if [[ -n "${ALLOW_PRIVILEGED}" ]]; then
      priv_arg="--allow-privileged "
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

    # Create client ca
    kube::util::create_signing_certkey "${CONTROLPLANE_SUDO}" "${CERT_DIR}" client '"client auth"'

    # Create client certs signed with client-ca, given id, given CN and a number of groups
    # NOTE: system:masters will be removed in the future
    kube::util::create_client_certkey "${CONTROLPLANE_SUDO}" "${CERT_DIR}" 'client-ca' kubelet system:node:${HOSTNAME_OVERRIDE} system:nodes
    kube::util::create_client_certkey "${CONTROLPLANE_SUDO}" "${CERT_DIR}" 'client-ca' kube-proxy system:kube-proxy system:nodes
    kube::util::create_client_certkey "${CONTROLPLANE_SUDO}" "${CERT_DIR}" 'client-ca' controller system:controller system:masters
    kube::util::create_client_certkey "${CONTROLPLANE_SUDO}" "${CERT_DIR}" 'client-ca' scheduler system:scheduler system:masters
    kube::util::create_client_certkey "${CONTROLPLANE_SUDO}" "${CERT_DIR}" 'client-ca' admin system:admin system:masters

    # Create auth proxy client ca
    kube::util::create_signing_certkey "${CONTROLPLANE_SUDO}" "${CERT_DIR}" request-header '"client auth"'
    kube::util::create_client_certkey "${CONTROLPLANE_SUDO}" "${CERT_DIR}" request-header-ca auth-proxy system:auth-proxy

    APISERVER_LOG=/tmp/kube-apiserver.log
    ${CONTROLPLANE_SUDO} "${GO_OUT}/hyperkube" apiserver ${anytoken_arg} ${authorizer_arg} ${priv_arg} ${runtime_config}\
      ${advertise_address} \
      --v=${LOG_LEVEL} \
      --cert-dir="${CERT_DIR}" \
      --client-ca-file="${CERT_DIR}/client-ca.crt" \
      --service-account-key-file="${SERVICE_ACCOUNT_KEY}" \
      --service-account-lookup="${SERVICE_ACCOUNT_LOOKUP}" \
      --admission-control="${ADMISSION_CONTROL}" \
      --bind-address="${API_BIND_ADDR}" \
      --secure-port="${API_SECURE_PORT}" \
      --tls-ca-file="${ROOT_CA_FILE}" \
      --insecure-bind-address="${API_HOST_IP}" \
      --insecure-port="${API_PORT}" \
      --etcd-servers="http://${ETCD_HOST}:${ETCD_PORT}" \
      --service-cluster-ip-range="${SERVICE_CLUSTER_IP_RANGE}" \
      --feature-gates="${FEATURE_GATES}" \
      --cloud-provider="${CLOUD_PROVIDER}" \
      --cloud-config="${CLOUD_CONFIG}" \
      --requestheader-username-headers=X-Remote-User \
      --requestheader-group-headers=X-Remote-Group \
      --requestheader-extra-headers-prefix=X-Remote-Extra- \
      --requestheader-client-ca-file="${CERT_DIR}/request-header-ca.crt" \
      --requestheader-allowed-names=system:auth-proxy \
      --cors-allowed-origins="${API_CORS_ALLOWED_ORIGINS}" >"${APISERVER_LOG}" 2>&1 &
    APISERVER_PID=$!

    # Wait for kube-apiserver to come up before launching the rest of the components.
    echo "Waiting for apiserver to come up"
    kube::util::wait_for_url "https://${API_HOST}:${API_SECURE_PORT}/version" "apiserver: " 1 ${WAIT_FOR_URL_API_SERVER} || exit 1

    # Create kubeconfigs for all components, using client certs
    kube::util::write_client_kubeconfig "${CONTROLPLANE_SUDO}" "${CERT_DIR}" "${ROOT_CA_FILE}" "${API_HOST}" "${API_SECURE_PORT}" admin
    kube::util::write_client_kubeconfig "${CONTROLPLANE_SUDO}" "${CERT_DIR}" "${ROOT_CA_FILE}" "${API_HOST}" "${API_SECURE_PORT}" kubelet
    kube::util::write_client_kubeconfig "${CONTROLPLANE_SUDO}" "${CERT_DIR}" "${ROOT_CA_FILE}" "${API_HOST}" "${API_SECURE_PORT}" kube-proxy
    kube::util::write_client_kubeconfig "${CONTROLPLANE_SUDO}" "${CERT_DIR}" "${ROOT_CA_FILE}" "${API_HOST}" "${API_SECURE_PORT}" controller
    kube::util::write_client_kubeconfig "${CONTROLPLANE_SUDO}" "${CERT_DIR}" "${ROOT_CA_FILE}" "${API_HOST}" "${API_SECURE_PORT}" scheduler

    if [[ -z "${AUTH_ARGS}"  ]]; then
        if [[ "${ALLOW_ANY_TOKEN}" = true  ]]; then
            # use token authentication
            if [[ -n "${KUBECONFIG_TOKEN}"  ]]; then
                AUTH_ARGS="--token=${KUBECONFIG_TOKEN}"
            else
                AUTH_ARGS="--token=system:admin/system:masters"
            fi
        else
            # default to use basic authentication
            AUTH_ARGS="--username=admin --password=admin"
        fi
    fi
}

function start_controller_manager {
    node_cidr_args=""
    if [[ "${NET_PLUGIN}" == "kubenet" ]]; then
      node_cidr_args="--allocate-node-cidrs=true --cluster-cidr=10.1.0.0/16 "
    fi

    CTLRMGR_LOG=/tmp/kube-controller-manager.log
    ${CONTROLPLANE_SUDO} "${GO_OUT}/hyperkube" controller-manager \
      --v=${LOG_LEVEL} \
      --service-account-private-key-file="${SERVICE_ACCOUNT_KEY}" \
      --root-ca-file="${ROOT_CA_FILE}" \
      --enable-hostpath-provisioner="${ENABLE_HOSTPATH_PROVISIONER}" \
      ${node_cidr_args} \
      --pvclaimbinder-sync-period="${CLAIM_BINDER_SYNC_PERIOD}" \
      --feature-gates="${FEATURE_GATES}" \
      --cloud-provider="${CLOUD_PROVIDER}" \
      --cloud-config="${CLOUD_CONFIG}" \
      --kubeconfig "$CERT_DIR"/controller.kubeconfig \
      --master="https://${API_HOST}:${API_SECURE_PORT}" >"${CTLRMGR_LOG}" 2>&1 &
    CTLRMGR_PID=$!
}

function start_kubelet {
    KUBELET_LOG=/tmp/kubelet.log

    priv_arg=""
    if [[ -n "${ALLOW_PRIVILEGED}" ]]; then
      priv_arg="--allow-privileged "
    fi

    mkdir -p /var/lib/kubelet
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

      net_plugin_dir_args=""
      if [[ -n "${NET_PLUGIN_DIR}" ]]; then
        net_plugin_dir_args="--network-plugin-dir=${NET_PLUGIN_DIR}"
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
        --chaos-chance="${CHAOS_CHANCE}" \
        --container-runtime="${CONTAINER_RUNTIME}" \
        --experimental-cri=${EXPERIMENTAL_CRI} \
        --rkt-path="${RKT_PATH}" \
        --rkt-stage1-image="${RKT_STAGE1_IMAGE}" \
        --hostname-override="${HOSTNAME_OVERRIDE}" \
        --cloud-provider="${CLOUD_PROVIDER}" \
        --cloud-config="${CLOUD_CONFIG}" \
        --address="${KUBELET_HOST}" \
        --require-kubeconfig \
        --kubeconfig "$CERT_DIR"/kubelet.kubeconfig \
        --feature-gates="${FEATURE_GATES}" \
        --cpu-cfs-quota=${CPU_CFS_QUOTA} \
        --enable-controller-attach-detach="${ENABLE_CONTROLLER_ATTACH_DETACH}" \
        --experimental-cgroups-per-qos=${EXPERIMENTAL_CGROUPS_PER_QOS} \
        --cgroup-driver=${CGROUP_DRIVER} \
        --cgroup-root=${CGROUP_ROOT} \
        ${auth_args} \
        ${dns_args} \
        ${net_plugin_dir_args} \
        ${net_plugin_args} \
        ${container_runtime_endpoint_args} \
        ${image_service_endpoint_args} \
        --port="$KUBELET_PORT" >"${KUBELET_LOG}" 2>&1 &
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
        ${cred_bind} \
        --net=host \
        --privileged=true \
        -i \
        --cidfile=$KUBELET_CIDFILE \
        gcr.io/google_containers/kubelet \
        /kubelet --v=${LOG_LEVEL} --containerized ${priv_arg}--chaos-chance="${CHAOS_CHANCE}" --hostname-override="${HOSTNAME_OVERRIDE}" --cloud-provider="${CLOUD_PROVIDER}" --cloud-config="${CLOUD_CONFIG}" \ --address="127.0.0.1" --require-kubeconfig --kubeconfig "$CERT_DIR"/kubelet.kubeconfig --api-servers="https://${API_HOST}:${API_SECURE_PORT}" --port="$KUBELET_PORT"  --enable-controller-attach-detach="${ENABLE_CONTROLLER_ATTACH_DETACH}" &> $KUBELET_LOG &
    fi
}

function start_kubeproxy {
    PROXY_LOG=/tmp/kube-proxy.log
    sudo "${GO_OUT}/hyperkube" proxy \
      --v=${LOG_LEVEL} \
      --hostname-override="${HOSTNAME_OVERRIDE}" \
      --feature-gates="${FEATURE_GATES}" \
      --kubeconfig "$CERT_DIR"/kube-proxy.kubeconfig \
      --master="https://${API_HOST}:${API_SECURE_PORT}" >"${PROXY_LOG}" 2>&1 &
    PROXY_PID=$!

    SCHEDULER_LOG=/tmp/kube-scheduler.log
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
        if [[ "${FEDERATION:-}" == "true" ]]; then
          FEDERATIONS_DOMAIN_MAP="${FEDERATIONS_DOMAIN_MAP:-}"
          if [[ -z "${FEDERATIONS_DOMAIN_MAP}" && -n "${FEDERATION_NAME:-}" && -n "${DNS_ZONE_NAME:-}" ]]; then
            FEDERATIONS_DOMAIN_MAP="${FEDERATION_NAME}=${DNS_ZONE_NAME}"
          fi
          if [[ -n "${FEDERATIONS_DOMAIN_MAP}" ]]; then
            sed -i -e "s/{{ pillar\['federations_domain_map'\] }}/- --federations=${FEDERATIONS_DOMAIN_MAP}/g" kubedns-deployment.yaml
          else
            sed -i -e "/{{ pillar\['federations_domain_map'\] }}/d" kubedns-deployment.yaml
          fi
        else
          sed -i -e "/{{ pillar\['federations_domain_map'\] }}/d" kubedns-deployment.yaml
        fi
        sed -e "s/{{ pillar\['dns_server'\] }}/${DNS_SERVER_IP}/g" "${KUBE_ROOT}/cluster/addons/dns/kubedns-svc.yaml.in" >| kubedns-svc.yaml
        
        # TODO update to dns role once we have one.
        ${KUBECTL} --kubeconfig="${CERT_DIR}/admin.kubeconfig" create clusterrolebinding system:kube-dns --clusterrole=cluster-admin --serviceaccount=kube-system:default
        # use kubectl to create kubedns deployment and service
        ${KUBECTL} --kubeconfig="${CERT_DIR}/admin.kubeconfig" --namespace=kube-system create -f kubedns-deployment.yaml
        ${KUBECTL} --kubeconfig="${CERT_DIR}/admin.kubeconfig" --namespace=kube-system create -f kubedns-svc.yaml
        echo "Kube-dns deployment and service successfully deployed."
        rm  kubedns-deployment.yaml kubedns-svc.yaml
    fi
}

function create_psp_policy {
    echo "Create podsecuritypolicy policies for RBAC."
    ${KUBECTL} --kubeconfig="${CERT_DIR}/admin.kubeconfig" create -f ${KUBE_ROOT}/examples/podsecuritypolicy/rbac/policies.yaml
    ${KUBECTL} --kubeconfig="${CERT_DIR}/admin.kubeconfig" create -f ${KUBE_ROOT}/examples/podsecuritypolicy/rbac/roles.yaml
    ${KUBECTL} --kubeconfig="${CERT_DIR}/admin.kubeconfig" create -f ${KUBE_ROOT}/examples/podsecuritypolicy/rbac/bindings.yaml
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
To start using your cluster, open up another terminal/tab and run:

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
kube::etcd::validate

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
kube::util::test_cfssl_installed

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
fi

if [[ "${START_MODE}" != "nokubelet" ]]; then
  start_kubelet
fi

if [[ -n "${PSP_ADMISSION}" && "${ENABLE_RBAC}" = true ]]; then
    create_psp_policy
fi

print_success

if [[ "${ENABLE_DAEMON}" = false ]]; then
   while true; do sleep 1; done
fi
