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

# required:
# KUBE_ROOT: path of the root of the Kubernetes reposiitory

# optional override
# FEDERATION_IMAGE_REPO_BASE: repo which federated images are tagged under (default gcr.io/google_containers)
# FEDERATION_NAMESPACE: name of the namespace will created for the federated components in the underlying cluster.
# KUBE_PLATFORM
# KUBE_ARCH
# KUBE_BUILD_STAGE

: "${KUBE_ROOT?Must set KUBE_ROOT env var}"

# Provides the $KUBERNETES_PROVIDER variable and detect-project function
source "${KUBE_ROOT}/cluster/kube-util.sh"

# If $FEDERATION_PUSH_REPO_BASE isn't set, then set the GCR registry name
# based on the detected project name for gce and gke providers.
FEDERATION_PUSH_REPO_BASE=${FEDERATION_PUSH_REPO_BASE:-}
if [[ -z "${FEDERATION_PUSH_REPO_BASE}" ]]; then
    if [[ "${KUBERNETES_PROVIDER}" == "gke" || "${KUBERNETES_PROVIDER}" == "gce" ]]; then
        # Populates $PROJECT
        detect-project
        if [[ ${PROJECT} == *':'* ]]; then
            echo "${PROJECT} contains ':' and can not be used as FEDERATION_PUSH_REPO_BASE. Please set FEDERATION_PUSH_REPO_BASE explicitly."
            exit 1
        fi
        FEDERATION_PUSH_REPO_BASE=gcr.io/${PROJECT}
    else
        echo "Must set FEDERATION_PUSH_REPO_BASE env var"
        exit 1
    fi
fi

FEDERATION_IMAGE_REPO_BASE=${FEDERATION_IMAGE_REPO_BASE:-'gcr.io/google_containers'}
FEDERATION_NAMESPACE=${FEDERATION_NAMESPACE:-federation}

KUBE_PLATFORM=${KUBE_PLATFORM:-linux}
KUBE_ARCH=${KUBE_ARCH:-amd64}
KUBE_BUILD_STAGE=${KUBE_BUILD_STAGE:-release-stage}

source "${KUBE_ROOT}/cluster/common.sh"
source "${KUBE_ROOT}/hack/lib/util.sh"

host_kubectl="${KUBE_ROOT}/cluster/kubectl.sh --namespace=${FEDERATION_NAMESPACE}"

# required:
# FEDERATION_PUSH_REPO_BASE: repo to which federated container images will be pushed

# Optional
# FEDERATION_IMAGE_TAG: reference and pull all federated images with this tag. Used for ci testing
function create-federation-api-objects {
(
    : "${FEDERATION_PUSH_REPO_BASE?Must set FEDERATION_PUSH_REPO_BASE env var}"
    export FEDERATION_APISERVER_DEPLOYMENT_NAME="federation-apiserver"
    export FEDERATION_APISERVER_IMAGE_REPO="${FEDERATION_PUSH_REPO_BASE}/federation-apiserver"
    export FEDERATION_APISERVER_IMAGE_TAG="${FEDERATION_IMAGE_TAG:-$(cat ${KUBE_ROOT}/_output/${KUBE_BUILD_STAGE}/server/${KUBE_PLATFORM}-${KUBE_ARCH}/kubernetes/server/bin/federation-apiserver.docker_tag)}"

    export FEDERATION_CONTROLLER_MANAGER_DEPLOYMENT_NAME="federation-controller-manager"
    export FEDERATION_CONTROLLER_MANAGER_IMAGE_REPO="${FEDERATION_PUSH_REPO_BASE}/federation-controller-manager"
    export FEDERATION_CONTROLLER_MANAGER_IMAGE_TAG="${FEDERATION_IMAGE_TAG:-$(cat ${KUBE_ROOT}/_output/${KUBE_BUILD_STAGE}/server/${KUBE_PLATFORM}-${KUBE_ARCH}/kubernetes/server/bin/federation-controller-manager.docker_tag)}"

    if [[ -z "${FEDERATION_DNS_PROVIDER:-}" ]]; then
      # Set the appropriate value based on cloud provider.
      if [[ "$KUBERNETES_PROVIDER" == "gce" || "${KUBERNETES_PROVIDER}" == "gke" ]]; then
        echo "setting dns provider to google-clouddns"
        export FEDERATION_DNS_PROVIDER="google-clouddns"
      elif [[ "${KUBERNETES_PROVIDER}" == "aws" ]]; then
        echo "setting dns provider to aws-route53"
        export FEDERATION_DNS_PROVIDER="aws-route53"
      else
        echo "Must set FEDERATION_DNS_PROVIDER env var"
        exit 1
      fi
    fi

    export FEDERATION_SERVICE_CIDR=${FEDERATION_SERVICE_CIDR:-"10.10.0.0/24"}

    #Only used for providers that require a nodeport service (vagrant for now)
    #We will use loadbalancer services where we can
    export FEDERATION_API_NODEPORT=32111
    export FEDERATION_NAMESPACE
    export FEDERATION_NAME="${FEDERATION_NAME:-federation}"
    export DNS_ZONE_NAME="${DNS_ZONE_NAME:-federation.example}"  # See https://tools.ietf.org/html/rfc2606

    template="go run ${KUBE_ROOT}/federation/cluster/template.go"

    FEDERATION_KUBECONFIG_PATH="${KUBE_ROOT}/federation/cluster/kubeconfig"

    federation_kubectl="${KUBE_ROOT}/cluster/kubectl.sh --context=federation-cluster --namespace=default"

    manifests_root="${KUBE_ROOT}/federation/manifests/"

    $template "${manifests_root}/federation-ns.yaml" | $host_kubectl apply -f -

    cleanup-federation-api-objects

    export FEDERATION_API_HOST=""
    export KUBE_MASTER_IP=""
    if [[ "$KUBERNETES_PROVIDER" == "vagrant" ]];then
	# The vagrant approach is to use a nodeport service, and point kubectl at one of the nodes
	$template "${manifests_root}/federation-apiserver-nodeport-service.yaml" | $host_kubectl create -f -
	node_addresses=`$host_kubectl get nodes -o=jsonpath='{.items[*].status.addresses[?(@.type=="InternalIP")].address}'`
	FEDERATION_API_HOST=`printf "$node_addresses" | cut -d " " -f1`
	KUBE_MASTER_IP="${FEDERATION_API_HOST}:${FEDERATION_API_NODEPORT}"
    elif [[ "$KUBERNETES_PROVIDER" == "gce" || "$KUBERNETES_PROVIDER" == "gke" || "$KUBERNETES_PROVIDER" == "aws" ]];then
	# any capable providers should use a loadbalancer service
	# we check for ingress.ip and ingress.hostname, so should work for any loadbalancer-providing provider
	# allows 30x5 = 150 seconds for loadbalancer creation
	$template "${manifests_root}/federation-apiserver-lb-service.yaml" | $host_kubectl create -f -
	for i in {1..30};do
	    echo "attempting to get federation-apiserver loadbalancer hostname ($i / 30)"
	    for field in ip hostname;do
		FEDERATION_API_HOST=`${host_kubectl} get -o=jsonpath svc/${FEDERATION_APISERVER_DEPLOYMENT_NAME} --template '{.status.loadBalancer.ingress[*].'"${field}}"`
		if [[ ! -z "${FEDERATION_API_HOST// }" ]];then
		    break 2
		fi
	    done
	    if [[ $i -eq 30 ]];then
		echo "Could not find ingress hostname for federation-apiserver loadbalancer service"
		exit 1
	    fi
	    sleep 5
	done
	KUBE_MASTER_IP="${FEDERATION_API_HOST}:443"
    else
	echo "provider ${KUBERNETES_PROVIDER} is not (yet) supported for e2e testing"
	exit 1
    fi
    echo "Found federation-apiserver host at $FEDERATION_API_HOST"

    FEDERATION_API_TOKEN="$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)"
    export FEDERATION_API_KNOWN_TOKENS="${FEDERATION_API_TOKEN},admin,admin"

    # Create a kubeconfig with credentails for federation-apiserver. We will
    # then use this kubeconfig to create a secret which the federation
    # controller manager can use to talk to the federation-apiserver.
    # Note that the file name should be "kubeconfig" so that the secret key gets the same name.
    KUBECONFIG_DIR=$(dirname ${KUBECONFIG:-$DEFAULT_KUBECONFIG})
    CONTEXT=federation-cluster \
	   KUBE_BEARER_TOKEN="$FEDERATION_API_TOKEN" \
           KUBECONFIG="${KUBECONFIG_DIR}/federation/federation-apiserver/kubeconfig" \
	   create-kubeconfig

    # Create secret with federation-apiserver's kubeconfig
    $host_kubectl create secret generic federation-apiserver-kubeconfig --from-file="${KUBECONFIG_DIR}/federation/federation-apiserver/kubeconfig" --namespace="${FEDERATION_NAMESPACE}"

    # Create secrets with all the kubernetes-apiserver's kubeconfigs.
    # Note: This is used only by the test setup (where kubernetes clusters are
    # brought up with FEDERATION=true). Users are expected to create this secret
    # themselves.
    for dir in ${KUBECONFIG_DIR}/federation/kubernetes-apiserver/*; do
      # We create a secret with the same name as the directory name (which is
      # same as cluster name in kubeconfig).
      # Massage the name so that it is valid (should not contain "_" and max 253
      # chars)
      name=$(basename $dir)
      name=$(echo "$name" | sed -e "s/_/-/g")  # Replace "_" by "-"
      name=${name:0:252}
      echo "Creating secret with name: $name"
      $host_kubectl create secret generic ${name} --from-file="${dir}/kubeconfig" --namespace="${FEDERATION_NAMESPACE}"
    done

    for file in federation-etcd-pvc.yaml federation-apiserver-{deployment,secrets}.yaml federation-controller-manager-deployment.yaml; do
      $template "${manifests_root}/${file}" | $host_kubectl create -f -
    done

    # Update the users kubeconfig to include federation-apiserver credentials.
    CONTEXT=federation-cluster \
	   KUBE_BEARER_TOKEN="$FEDERATION_API_TOKEN" \
	   SECONDARY_KUBECONFIG=true \
	   create-kubeconfig

    # Don't finish provisioning until federation-apiserver pod is running
    for i in {1..30};do
	#TODO(colhom): in the future this needs to scale out for N pods. This assumes just one pod
	phase="$($host_kubectl get -o=jsonpath pods -lapp=federated-cluster,module=federation-apiserver --template '{.items[*].status.phase}')"
	echo "Waiting for federation-apiserver to be running...(phase= $phase)"
	if [[ "$phase" == "Running" ]];then
	    echo "federation-apiserver pod is running!"
	    break
	fi

	if [[ $i -eq 30 ]];then
	    echo "federation-apiserver pod is not running! giving up."
	    exit 1
	fi

	sleep 4
    done

    # Verify that federation-controller-manager pod is running.
    for i in {1..30};do
	#TODO(colhom): in the future this needs to scale out for N pods. This assumes just one pod
	phase="$($host_kubectl get -o=jsonpath pods -lapp=federated-cluster,module=federation-controller-manager --template '{.items[*].status.phase}')"
	echo "Waiting for federation-controller-manager to be running...(phase= $phase)"
	if [[ "$phase" == "Running" ]];then
	    echo "federation-controller-manager pod is running!"
	    break
	fi

	if [[ $i -eq 30 ]];then
	    echo "federation-controller-manager pod is not running! giving up."
	    exit 1
	fi

	sleep 4
    done
)
}

# Required
# FEDERATION_PUSH_REPO_BASE: the docker repo where federated images will be pushed

# Optional
# FEDERATION_IMAGE_TAG: push all federated images with this tag. Used for ci testing
function push-federation-images {
    : "${FEDERATION_PUSH_REPO_BASE?Must set FEDERATION_PUSH_REPO_BASE env var}"
    local FEDERATION_BINARIES=${FEDERATION_BINARIES:-"federation-apiserver federation-controller-manager"}

    local imageFolder="${KUBE_ROOT}/_output/${KUBE_BUILD_STAGE}/server/${KUBE_PLATFORM}-${KUBE_ARCH}/kubernetes/server/bin"

    if [[ ! -d "$imageFolder" ]];then
	echo "${imageFolder} does not exist! Run make quick-release or make release"
	exit 1
    fi

    for binary in $FEDERATION_BINARIES;do
	local imageFile="${imageFolder}/${binary}.tar"

	if [[ ! -f "$imageFile" ]];then
	    echo "${imageFile} does not exist!"
	    exit 1
	fi

	echo "Load: ${imageFile}"
	# Load the image. Trust we know what it's called, as docker load provides no help there :(
	docker load < "${imageFile}"

	local srcImageTag="$(cat ${imageFolder}/${binary}.docker_tag)"
	local dstImageTag="${FEDERATION_IMAGE_TAG:-$srcImageTag}"
	local srcImageName="${FEDERATION_IMAGE_REPO_BASE}/${binary}:${srcImageTag}"
	local dstImageName="${FEDERATION_PUSH_REPO_BASE}/${binary}:${dstImageTag}"

	echo "Tag: ${srcImageName} --> ${dstImageName}"
	docker tag -f "$srcImageName" "$dstImageName"

	echo "Push: $dstImageName"
	if [[ "${FEDERATION_PUSH_REPO_BASE}" == "gcr.io/"* ]];then
	    echo " -> GCR repository detected. Using gcloud"
	    gcloud docker push "$dstImageName"
	else
	    docker push "$dstImageName"
	fi

	echo "Remove: $srcImageName"
	docker rmi "$srcImageName"

	if [[ "$srcImageName" != "dstImageName" ]];then
	    echo "Remove: $dstImageName"
	    docker rmi "$dstImageName"
	fi

    done
}
function cleanup-federation-api-objects {
  # Delete all resources with the federated-cluster label.
  $host_kubectl delete pods,svc,rc,deployment,secret -lapp=federated-cluster
  # Delete all resources in FEDERATION_NAMESPACE.
  $host_kubectl delete pods,svc,rc,deployment,secret --namespace=${FEDERATION_NAMESPACE} --all
}
