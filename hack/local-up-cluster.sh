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

# This command builds and runs a local kubernetes cluster. It's just like
# local-up.sh, but this one launches the three separate binaries.
# You may need to run this as root to allow kubelet to open docker's socket.
DOCKER_OPTS=${DOCKER_OPTS:-""}
DOCKER_NATIVE=${DOCKER_NATIVE:-""}
DOCKER=(docker ${DOCKER_OPTS})
DOCKERIZE_KUBELET=${DOCKERIZE_KUBELET:-""}
ALLOW_PRIVILEGED=${ALLOW_PRIVILEGED:-""}
ALLOW_SECURITY_CONTEXT=${ALLOW_SECURITY_CONTEXT:-""}
RUNTIME_CONFIG=${RUNTIME_CONFIG:-""}
NET_PLUGIN=${NET_PLUGIN:-""}
NET_PLUGIN_DIR=${NET_PLUGIN_DIR:-""}
KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
# We disable cluster DNS by default because this script uses docker0 (or whatever
# container bridge docker is currently using) and we don't know the IP of the
# DNS pod to pass in as --cluster-dns. To set this up by hand, set this flag
# and change DNS_SERVER_IP to the appropriate IP.
ENABLE_CLUSTER_DNS=${KUBE_ENABLE_CLUSTER_DNS:-false}
DNS_SERVER_IP=${KUBE_DNS_SERVER_IP:-10.0.0.10}
DNS_DOMAIN=${KUBE_DNS_NAME:-"cluster.local"}
DNS_REPLICAS=${KUBE_DNS_REPLICAS:-1}
KUBECTL=${KUBECTL:-cluster/kubectl.sh}
WAIT_FOR_URL_API_SERVER=${WAIT_FOR_URL_API_SERVER:-10}
ENABLE_DAEMON=${ENABLE_DAEMON:-false}
HOSTNAME_OVERRIDE=${HOSTNAME_OVERRIDE:-"127.0.0.1"}
CLOUD_PROVIDER=${CLOUD_PROVIDER:-""}
CLOUD_CONFIG=${CLOUD_CONFIG:-""}

# sanity check for OpenStack provider
if [ "${CLOUD_PROVIDER}" == "openstack" ]; then
    if [ "${CLOUD_CONFIG}" == "" ]; then
        echo "Missing CLOUD_CONFIG env for OpenStack provider!"
        exit 1
    fi
    if [ ! -f "${CLOUD_CONFIG}" ]; then
        echo "Cloud config ${CLOUD_CONFIG} doesn't exit"
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
            echo "Example 1: hack/local-up-cluster.sh -o _output/dockerized/bin/linux/amd64/ (run from docker output)"
            echo "Example 2: hack/local-up-cluster.sh (build a local copy of the source)"
}

### Allow user to supply the source directory.
GO_OUT=""
while getopts "o:" OPTION
do
    case $OPTION in
        o)
            echo "skipping build"
            echo "using source $OPTARG"
            GO_OUT="$OPTARG"
            if [ $GO_OUT == "" ]; then
                echo "You provided an invalid value for the build output directory."
                exit
            fi
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

function test_docker {
    ${DOCKER[@]} ps 2> /dev/null 1> /dev/null
    if [ "$?" != "0" ]; then
      echo "Failed to successfully run 'docker ps', please verify that docker is installed and \$DOCKER_HOST is set correctly."
      exit 1
    fi
}

function test_openssl_installed {
    openssl version >& /dev/null
    if [ "$?" != "0" ]; then
      echo "Failed to run openssl. Please ensure openssl is installed"
      exit 1
    fi
}

# Shut down anyway if there's an error.
set +e

API_PORT=${API_PORT:-8080}
API_HOST=${API_HOST:-127.0.0.1}
API_HOST_IP=${API_HOST_IP:-${API_HOST}}
API_BIND_ADDR=${API_HOST_IP:-"0.0.0.0"}
KUBELET_HOST=${KUBELET_HOST:-"127.0.0.1"}
# By default only allow CORS for requests on localhost
API_CORS_ALLOWED_ORIGINS=${API_CORS_ALLOWED_ORIGINS:-"/127.0.0.1(:[0-9]+)?$,/localhost(:[0-9]+)?$"}
KUBELET_PORT=${KUBELET_PORT:-10250}
LOG_LEVEL=${LOG_LEVEL:-3}
CONTAINER_RUNTIME=${CONTAINER_RUNTIME:-"docker"}
RKT_PATH=${RKT_PATH:-""}
RKT_STAGE1_IMAGE=${RKT_STAGE1_IMAGE:-""}
CHAOS_CHANCE=${CHAOS_CHANCE:-0.0}
CPU_CFS_QUOTA=${CPU_CFS_QUOTA:-false}
ENABLE_HOSTPATH_PROVISIONER=${ENABLE_HOSTPATH_PROVISIONER:-"false"}
CLAIM_BINDER_SYNC_PERIOD=${CLAIM_BINDER_SYNC_PERIOD:-"15s"} # current k8s default

function test_apiserver_off {
    # For the common local scenario, fail fast if server is already running.
    # this can happen if you run local-up-cluster.sh twice and kill etcd in between.
    curl --silent -g $API_HOST:$API_PORT
    if [ ! $? -eq 0 ]; then
        echo "API SERVER port is free, proceeding..."
    else
        echo "ERROR starting API SERVER, exiting.  Some host on $API_HOST is serving already on $API_PORT"
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

function startETCD {
    echo "Starting etcd"
    kube::etcd::start
}

function set_service_accounts {
    SERVICE_ACCOUNT_LOOKUP=${SERVICE_ACCOUNT_LOOKUP:-false}
    SERVICE_ACCOUNT_KEY=${SERVICE_ACCOUNT_KEY:-"/tmp/kube-serviceaccount.key"}
    # Generate ServiceAccount key if needed
    if [[ ! -f "${SERVICE_ACCOUNT_KEY}" ]]; then
      mkdir -p "$(dirname ${SERVICE_ACCOUNT_KEY})"
      openssl genrsa -out "${SERVICE_ACCOUNT_KEY}" 2048 2>/dev/null
    fi
}

function start_apiserver {
    # Admission Controllers to invoke prior to persisting objects in cluster
    if [[ -z "${ALLOW_SECURITY_CONTEXT}" ]]; then
      ADMISSION_CONTROL=NamespaceLifecycle,LimitRanger,SecurityContextDeny,ServiceAccount,ResourceQuota
    else
      ADMISSION_CONTROL=NamespaceLifecycle,LimitRanger,ServiceAccount,ResourceQuota
    fi
    # This is the default dir and filename where the apiserver will generate a self-signed cert
    # which should be able to be used as the CA to verify itself
    CERT_DIR=/var/run/kubernetes
    ROOT_CA_FILE=$CERT_DIR/apiserver.crt

    priv_arg=""
    if [[ -n "${ALLOW_PRIVILEGED}" ]]; then
      priv_arg="--allow-privileged "
    fi
    runtime_config=""
    if [[ -n "${RUNTIME_CONFIG}" ]]; then
      runtime_config="--runtime-config=${RUNTIME_CONFIG}"
    fi

    # Let the API server pick a default address when API_HOST
    # is set to 127.0.0.1
    advertise_address=""
    if [[ "${API_HOST}" != "127.0.0.1" ]]; then
        advertise_address="--advertise_address=${API_HOST}"
    fi

    APISERVER_LOG=/tmp/kube-apiserver.log
    sudo -E "${GO_OUT}/hyperkube" apiserver ${priv_arg} ${runtime_config}\
      ${advertise_address} \
      --v=${LOG_LEVEL} \
      --cert-dir="${CERT_DIR}" \
      --service-account-key-file="${SERVICE_ACCOUNT_KEY}" \
      --service-account-lookup="${SERVICE_ACCOUNT_LOOKUP}" \
      --admission-control="${ADMISSION_CONTROL}" \
      --bind-address="${API_BIND_ADDR}" \
      --insecure-bind-address="${API_HOST_IP}" \
      --insecure-port="${API_PORT}" \
      --etcd-servers="http://${ETCD_HOST}:${ETCD_PORT}" \
      --service-cluster-ip-range="10.0.0.0/24" \
      --cloud-provider="${CLOUD_PROVIDER}" \
      --cloud-config="${CLOUD_CONFIG}" \
      --cors-allowed-origins="${API_CORS_ALLOWED_ORIGINS}" >"${APISERVER_LOG}" 2>&1 &
    APISERVER_PID=$!

    # Wait for kube-apiserver to come up before launching the rest of the components.
    echo "Waiting for apiserver to come up"
    kube::util::wait_for_url "http://${API_HOST}:${API_PORT}/api/v1/pods" "apiserver: " 1 ${WAIT_FOR_URL_API_SERVER} || exit 1
}

function start_controller_manager {
    node_cidr_args=""
    if [[ "${NET_PLUGIN}" == "kubenet" ]]; then
      node_cidr_args="--allocate-node-cidrs=true --cluster-cidr=10.1.0.0/16 "
    fi

    CTLRMGR_LOG=/tmp/kube-controller-manager.log
    sudo -E "${GO_OUT}/hyperkube" controller-manager \
      --v=${LOG_LEVEL} \
      --service-account-private-key-file="${SERVICE_ACCOUNT_KEY}" \
      --root-ca-file="${ROOT_CA_FILE}" \
      --enable-hostpath-provisioner="${ENABLE_HOSTPATH_PROVISIONER}" \
      ${node_cidr_args} \
      --pvclaimbinder-sync-period="${CLAIM_BINDER_SYNC_PERIOD}" \
      --cloud-provider="${CLOUD_PROVIDER}" \
      --cloud-config="${CLOUD_CONFIG}" \
      --master="${API_HOST}:${API_PORT}" >"${CTLRMGR_LOG}" 2>&1 &
    CTLRMGR_PID=$!
}

function start_kubelet {
    KUBELET_LOG=/tmp/kubelet.log

    mkdir -p /var/lib/kubelet
    if [[ -z "${DOCKERIZE_KUBELET}" ]]; then
      # On selinux enabled systems, it might
      # require to relabel /var/lib/kubelet
      if which selinuxenabled &> /dev/null && \
         selinuxenabled && \
         which chcon > /dev/null ; then
         if [[ ! $(ls -Zd /var/lib/kubelet) =~ system_u:object_r:svirt_sandbox_file_t:s0 ]] ; then
            echo "Applying SELinux label to /var/lib/kubelet directory."
            if ! sudo chcon -Rt svirt_sandbox_file_t /var/lib/kubelet; then
               echo "Failed to apply selinux label to /var/lib/kubelet."
            fi
         fi
      fi
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
      
      net_plugin_dir_args=""
      if [[ -n "${NET_PLUGIN_DIR}" ]]; then
        net_plugin_dir_args="--network-plugin-dir=${NET_PLUGIN_DIR}"
      fi

      kubenet_plugin_args=""
      if [[ "${NET_PLUGIN}" == "kubenet" ]]; then
        kubenet_plugin_args="--reconcile-cidr=true "
      fi

      sudo -E "${GO_OUT}/hyperkube" kubelet ${priv_arg}\
        --v=${LOG_LEVEL} \
        --chaos-chance="${CHAOS_CHANCE}" \
        --container-runtime="${CONTAINER_RUNTIME}" \
        --rkt-path="${RKT_PATH}" \
        --rkt-stage1-image="${RKT_STAGE1_IMAGE}" \
        --hostname-override="${HOSTNAME_OVERRIDE}" \
        --cloud-provider="${CLOUD_PROVIDER}" \
        --cloud-config="${CLOUD_CONFIG}" \
        --address="${KUBELET_HOST}" \
        --api-servers="${API_HOST}:${API_PORT}" \
        --cpu-cfs-quota=${CPU_CFS_QUOTA} \
        ${dns_args} \
        ${net_plugin_dir_args} \
        ${net_plugin_args} \
        ${kubenet_plugin_args} \
        --port="$KUBELET_PORT" >"${KUBELET_LOG}" 2>&1 &
      KUBELET_PID=$!
    else
      # Docker won't run a container with a cidfile (container id file)
      # unless that file does not already exist; clean up an existing
      # dockerized kubelet that might be running.
      cleanup_dockerized_kubelet
      cred_bind=""
      # path to cloud credentails. 
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
        --volume=/var/lib/kubelet/:/var/lib/kubelet:rw,z \
        --volume=/dev:/dev \
        ${cred_bind} \
        --net=host \
        --privileged=true \
        -i \
        --cidfile=$KUBELET_CIDFILE \
        gcr.io/google_containers/kubelet \
        /kubelet --v=${LOG_LEVEL} --containerized ${priv_arg}--chaos-chance="${CHAOS_CHANCE}" --hostname-override="${HOSTNAME_OVERRIDE}" --cloud-provider="${CLOUD_PROVIDER}" --cloud-config="${CLOUD_CONFIG}" \ --address="127.0.0.1" --api-servers="${API_HOST}:${API_PORT}" --port="$KUBELET_PORT"  &> $KUBELET_LOG &
    fi
}

function start_kubeproxy {
    PROXY_LOG=/tmp/kube-proxy.log
    sudo -E "${GO_OUT}/hyperkube" proxy \
      --v=${LOG_LEVEL} \
      --hostname-override="${HOSTNAME_OVERRIDE}" \
      --master="http://${API_HOST}:${API_PORT}" >"${PROXY_LOG}" 2>&1 &
    PROXY_PID=$!

    SCHEDULER_LOG=/tmp/kube-scheduler.log
    sudo -E "${GO_OUT}/hyperkube" scheduler \
      --v=${LOG_LEVEL} \
      --master="http://${API_HOST}:${API_PORT}" >"${SCHEDULER_LOG}" 2>&1 &
    SCHEDULER_PID=$!
}

function start_kubedns {

    if [[ "${ENABLE_CLUSTER_DNS}" = true ]]; then
        echo "Creating kube-system namespace"
        sed -e "s/{{ pillar\['dns_replicas'\] }}/${DNS_REPLICAS}/g;s/{{ pillar\['dns_domain'\] }}/${DNS_DOMAIN}/g;" "${KUBE_ROOT}/cluster/addons/dns/skydns-rc.yaml.in" >| skydns-rc.yaml
        if [[ "${FEDERATION:-}" == "true" ]]; then
          FEDERATIONS_DOMAIN_MAP="${FEDERATIONS_DOMAIN_MAP:-}"
          if [[ -z "${FEDERATIONS_DOMAIN_MAP}" && -n "${FEDERATION_NAME:-}" && -n "${DNS_ZONE_NAME:-}" ]]; then
            FEDERATIONS_DOMAIN_MAP="${FEDERATION_NAME}=${DNS_ZONE_NAME}"
          fi
          if [[ -n "${FEDERATIONS_DOMAIN_MAP}" ]]; then
            sed -i -e "s/{{ pillar\['federations_domain_map'\] }}/- --federations=${FEDERATIONS_DOMAIN_MAP}/g" skydns-rc.yaml
          else
            sed -i -e "/{{ pillar\['federations_domain_map'\] }}/d" skydns-rc.yaml
          fi
        else
          sed -i -e "/{{ pillar\['federations_domain_map'\] }}/d" skydns-rc.yaml
        fi
        sed -e "s/{{ pillar\['dns_server'\] }}/${DNS_SERVER_IP}/g" "${KUBE_ROOT}/cluster/addons/dns/skydns-svc.yaml.in" >| skydns-svc.yaml
        cat <<EOF >namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: kube-system
EOF
        ${KUBECTL} config set-cluster local --server=http://${API_HOST}:${API_PORT} --insecure-skip-tls-verify=true
        ${KUBECTL} config set-context local --cluster=local
        ${KUBECTL} config use-context local

        ${KUBECTL} create -f namespace.yaml
        # use kubectl to create skydns rc and service
        ${KUBECTL} --namespace=kube-system create -f skydns-rc.yaml
        ${KUBECTL} --namespace=kube-system create -f skydns-svc.yaml
        echo "Kube-dns rc and service successfully deployed."
    fi

}

function print_success {
cat <<EOF
Local Kubernetes cluster is running. Press Ctrl-C to shut it down.

Logs:
  ${APISERVER_LOG}
  ${CTLRMGR_LOG}
  ${PROXY_LOG}
  ${SCHEDULER_LOG}
  ${KUBELET_LOG}

To start using your cluster, open up another terminal/tab and run:

  export KUBERNETES_PROVIDER=local

  cluster/kubectl.sh config set-cluster local --server=http://${API_HOST}:${API_PORT} --insecure-skip-tls-verify=true
  cluster/kubectl.sh config set-context local --cluster=local
  cluster/kubectl.sh config use-context local
  cluster/kubectl.sh
EOF
}

test_docker
test_apiserver_off
test_openssl_installed

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
startETCD
set_service_accounts
start_apiserver
start_controller_manager
start_kubelet
start_kubeproxy
start_kubedns
print_success

if [[ "${ENABLE_DAEMON}" = false ]]; then
   while true; do sleep 1; done
fi
