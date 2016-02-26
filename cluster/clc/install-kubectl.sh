#/usr/bin/env bash

function exit_message() {
    echo "ERROR: $1" >&2
    exit 1
}

if [ -z ${CLC_CLUSTER_NAME+null_if_undefined} ]
then
  exit_message "please define environment variable CLC_CLUSTER_NAME"
fi

CLC_CLUSTER_HOME=${HOME}/.clc_kube/${CLC_CLUSTER_NAME}

if [ ! -d ${CLC_CLUSTER_HOME} ]
then
  exit_message "cluster config directory ${CLC_CLUSTER_HOME} does not exist"
fi

HOSTS=${CLC_CLUSTER_HOME}/hosts
if [ ! -d  ${HOSTS} ]
then
  exit_message "ansible hosts directory ${HOSTS} does not exist"
fi

PKI=${CLC_CLUSTER_HOME}/pki
if [ ! -d  ${PKI} ]
then
  exit_message "public key infrastructure directory ${PKI} does not exist"
fi

### installing kubectl

KUBECTL=$(which kubectl)
if [ -z "$KUBECTL" ]
then
  version=v1.1.7
  arch=$(uname -s | tr '[:upper:]' '[:lower:]')  # linux|darwin
  url="https://storage.googleapis.com/kubernetes-release/release/${version}/bin/${arch}/amd64/kubectl"
  echo "No kubectl found, installing kubectl $version $arch to /usr/local/bin"
  curl -s -O $url
  chmod a+x kubectl
  mv kubectl /usr/local/bin
  KUBECTL=/usr/local/bin/kubectl
fi

echo kubectl binary is located at $KUBECTL
$KUBECTL version -c

### configuring kubectl
set -e

K8S_CLUSTER=${K8S_CLUSTER-$CLC_CLUSTER_NAME}
K8S_USER=${K8S_USER-admin}
K8S_NS=${K8S_NS-default}

# extract master ip from hosts file
MASTER_IP=$(grep -A1 master ${HOSTS}/hosts-* | grep -oE "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b")
SECURE_PORT=6443

# set default kube config file location to local file kubecfg_${K8S_CLUSTER}
OLDKUBECONFIG=${KUBECONFIG-~/.kube/config}
mkdir -p ${CLC_CLUSTER_HOME}/kube
export KUBECONFIG="${CLC_CLUSTER_HOME}/kube/config"

# set cluster
kubectl config set-cluster ${K8S_CLUSTER} \
   --server https://${MASTER_IP}:${SECURE_PORT} \
   --insecure-skip-tls-verify=false \
   --embed-certs=true \
   --certificate-authority=${PKI}/ca.crt

# user, credentials (reusing the kubelet/kube-proxy certificate)
kubectl config set-credentials ${K8S_USER}/${K8S_CLUSTER} \
   --embed-certs=true \
   --client-certificate=${PKI}/kubecfg.crt \
   --client-key=${PKI}/kubecfg.key

# define context
kubectl config set-context ${K8S_NS}/${K8S_CLUSTER}/${K8S_USER} \
    --user=${K8S_USER}/${K8S_CLUSTER} \
    --namespace=${K8S_NS} \
    --cluster=${K8S_CLUSTER}

# use context
kubectl config use-context ${K8S_NS}/${K8S_CLUSTER}/${K8S_USER}

# test
kubectl --kubeconfig=${KUBECONFIG} cluster-info

cat << MESSAGE >&1
in order to use kubectl to talk to the cluster, set the
environment variable KUBECONFIG with

   export KUBECONFIG="${KUBECONFIG}"

MESSAGE
