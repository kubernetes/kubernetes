#!/usr/bin/env bash

# Copyright 2018 The Kubernetes Authors.
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

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/../../..

# Creates a new kube-spawn cluster
function create-clusters {
	# shellcheck disable=SC2154	# Color defined in sourced script
	echo -e "${color_yellow}CHECKING CLUSTERS${color_norm}"
	if ibmcloud ks clusters | grep -Fq 'deleting'; then
		echo -n "Deleting old clusters"
	fi
	while ibmcloud ks clusters | grep -Fq 'deleting'
	do
		echo -n "."
		sleep 10
	done
	echo ""
	ibmcloud ks region-set us-east >/dev/null
	ibmcloud ks vlans wdc06 >/dev/null
	PRIVLAN=$(ibmcloud ks vlans wdc06 --json | jq '. | .[] | select(.type == "private") | .id' | sed -e "s/\"//g")
	PUBVLAN=$(ibmcloud ks vlans wdc06 --json | jq '. | .[] | select(.type == "public") | .id' | sed -e "s/\"//g")
	if ! ibmcloud ks clusters | grep -Fq 'kubeSpawnTester'; then
		echo "Creating spawning cluster"
		# make number and spec of node workers configurable
		# otherwise it can't afford tests like kubemark-5000
		# TODO: dynamically adjust the number and spec
		ibmcloud ks cluster-create --location "${CLUSTER_LOCATION}" --public-vlan "${PUBVLAN}" --private-vlan "${PRIVLAN}" --workers "${NUM_NODES:-2}" --machine-type "${NODE_SIZE}" --name kubeSpawnTester
	fi
	if ! ibmcloud ks clusters | grep -Fq 'kubeMasterTester'; then
		echo "Creating master cluster"
		# if we can't make it a bare master (workers = 0)
		# then make workers = 1 with the smallest machine spec
		ibmcloud ks cluster-create --location "${CLUSTER_LOCATION}" --public-vlan "${PUBVLAN}" --private-vlan "${PRIVLAN}" --workers 1 --machine-type u2c.2x4 --name kubeMasterTester
	fi
	push-image
	if ! ibmcloud ks clusters | grep 'kubeSpawnTester' | grep -Fq 'normal'; then
		# shellcheck disable=SC2154 # Color defined in sourced script
		echo -e "${color_cyan}Warning: new clusters may take up to 60 minutes to be ready${color_norm}"
		echo -n "Clusters loading"
	fi
	while ! ibmcloud ks clusters | grep 'kubeSpawnTester' | grep -Fq 'normal'
	do
		echo -n "."
		sleep 5
	done
	while ! ibmcloud ks clusters | grep 'kubeMasterTester' | grep -Fq 'normal'
	do
		echo -n "."
		sleep 5
	done
	echo -e "${color_yellow}CLUSTER CREATION COMPLETE${color_norm}"
}

# Builds and pushes image to registry
function push-image {
	if [[ "${ISBUILD}" = "y" ]]; then
		if ! ibmcloud cr namespaces | grep -Fq "${KUBE_NAMESPACE}"; then
			echo "Creating registry namespace"
			ibmcloud cr namespace-add "${KUBE_NAMESPACE}"
			echo "ibmcloud cr namespace-rm ${KUBE_NAMESPACE}" >> "${RESOURCE_DIRECTORY}/iks-namespacelist.sh"
		fi
		docker build -t "${KUBEMARK_INIT_TAG}" "${KUBEMARK_IMAGE_LOCATION}"
		docker tag "${KUBEMARK_INIT_TAG}" "${KUBEMARK_IMAGE_REGISTRY}${KUBE_NAMESPACE}/${PROJECT}:${KUBEMARK_IMAGE_TAG}"
		docker push "${KUBEMARK_IMAGE_REGISTRY}${KUBE_NAMESPACE}/${PROJECT}:${KUBEMARK_IMAGE_TAG}"
		echo "Image pushed"
	else
		KUBEMARK_IMAGE_REGISTRY="${KUBEMARK_IMAGE_REGISTRY:-brandondr96}"
		KUBE_NAMESPACE=""
	fi
}

# Allow user to use existing clusters if desired
function choose-clusters {
	echo -n -e "Do you want to use custom clusters? [y/N]${color_cyan}>${color_norm} "
	read -r USE_EXISTING
	if [[ "${USE_EXISTING}" = "y" ]]; then
		echo -e "${color_yellow}Enter path for desired hollow-node spawning cluster kubeconfig file:${color_norm}"
		read -r CUSTOM_SPAWN_CONFIG
		echo -e "${color_yellow}Enter path for desired hollow-node hosting cluster kubeconfig file:${color_norm}"
		read -r CUSTOM_MASTER_CONFIG
		push-image
	elif [[ "${USE_EXISTING}" = "N" ]]; then
		create-clusters
	else
		# shellcheck disable=SC2154 # Color defined in sourced script
		echo -e "${color_red}Invalid response, please try again:${color_norm}"
		choose-clusters
	fi
}

# Ensure secrets are correctly set
function set-registry-secrets {
	spawn-config
	kubectl get secret bluemix-default-secret-regional -o yaml | sed 's/default/kubemark/g' | kubectl -n kubemark create -f -
	kubectl patch serviceaccount -n kubemark default -p '{"imagePullSecrets": [{"name": "bluemix-kubemark-secret"}]}'
	kubectl -n kubemark get serviceaccounts default -o json | jq 'del(.metadata.resourceVersion)' | jq 'setpath(["imagePullSecrets"];[{"name":"bluemix-kubemark-secret-regional"}])' | kubectl -n kubemark replace serviceaccount default -f -
}

# Sets the hollow-node master
# Exported variables:
#   MASTER_IP - IP Address of the Kubemark master
function set-hollow-master {
	echo -e "${color_yellow}CONFIGURING MASTER${color_norm}"
	master-config
	MASTER_IP=$(grep server "$KUBECONFIG" | awk -F "/" '{print $3}')
	export MASTER_IP
}

# Set up master cluster environment
# Exported variables:
#   KUBECONFIG - Overrides default kube config for the purpose of setting up the Kubemark master components.
function master-config {
	if [[ "${USE_EXISTING}" = "y" ]]; then
		export KUBECONFIG=${CUSTOM_MASTER_CONFIG}
	else
		eval "$(ibmcloud ks cluster-config kubeMasterTester --admin | grep export)"
	fi
}

# Set up spawn cluster environment
# Exported variables:
#    KUBECONFIG - Overrides default kube config for the purpose of setting up the hollow-node cluster.
function spawn-config {
	if [[ "${USE_EXISTING}" = "y" ]]; then
		export KUBECONFIG=${CUSTOM_SPAWN_CONFIG}
	else
		eval "$(ibmcloud ks cluster-config kubeSpawnTester --admin | grep export)"
	fi
}

# Deletes existing clusters
function delete-clusters {
	echo "DELETING CLUSTERS"
	ibmcloud ks cluster-rm kubeSpawnTester
	ibmcloud ks cluster-rm kubeMasterTester
	while ! ibmcloud ks clusters | grep 'kubeSpawnTester' | grep -Fq 'deleting'
	do
		sleep 5
	done
	while ! ibmcloud ks clusters | grep 'kubeMasterTester' | grep -Fq 'deleting'
	do
		sleep 5
	done
	kubectl delete ns kubemark
}

# Login to cloud services
function complete-login {
	echo -e "${color_yellow}LOGGING INTO CLOUD SERVICES${color_norm}"
	echo -n -e "Do you have a federated IBM cloud login? [y/N]${color_cyan}>${color_norm} "
	read -r ISFED
	if [[ "${ISFED}" = "y" ]]; then
		ibmcloud login --sso -a "${REGISTRY_LOGIN_URL}"
	elif [[ "${ISFED}" = "N" ]]; then
		ibmcloud login -a "${REGISTRY_LOGIN_URL}"
	else
		echo -e "${color_red}Invalid response, please try again:${color_norm}"
		complete-login
	fi
	ibmcloud cr login
}

# Generate values to fill the hollow-node configuration templates.
# Exported variables:
#   KUBECTL - The name or path to the kubernetes client binary.
#   TEST_CLUSTER_API_CONTENT_TYPE - Defines the content-type of the requests used by the Kubemark components.
function generate-values {
	echo "Generating values"
	master-config
	KUBECTL=kubectl
	export KUBECTL
	KUBEMARK_DIRECTORY="${KUBE_ROOT}/test/kubemark"
	RESOURCE_DIRECTORY="${KUBEMARK_DIRECTORY}/resources"
	TEST_CLUSTER_API_CONTENT_TYPE="bluemix" #Determine correct usage of this
	export TEST_CLUSTER_API_CONTENT_TYPE
	CONFIGPATH=${KUBECONFIG%/*}
	KUBELET_CERT_BASE64="${KUBELET_CERT_BASE64:-$(base64 "${CONFIGPATH}/admin.pem" | tr -d '\r\n')}"
	KUBELET_KEY_BASE64="${KUBELET_KEY_BASE64:-$(base64 "${CONFIGPATH}/admin-key.pem" | tr -d '\r\n')}"
	CA_CERT_BASE64="${CA_CERT_BASE64:-$( base64 "$(find "${CONFIGPATH}" -name "*ca*" | head -n 1)" | tr -d '\r\n')}"

}

# Build image for kubemark
function build-kubemark-image {
	echo -n -e "Do you want to build the kubemark image? [y/N]${color_cyan}>${color_norm} "
	read -r ISBUILD
	if [[ "${ISBUILD}" = "y" ]]; then
		echo -e "${color_yellow}BUILDING IMAGE${color_norm}"
		"${KUBE_ROOT}/build/run.sh" make kubemark
		cp "${KUBE_ROOT}/_output/dockerized/bin/linux/amd64/kubemark" "${KUBEMARK_IMAGE_LOCATION}"
	elif [[ "${ISBUILD}" = "N" ]]; then
		echo -n ""
	else
		echo -e "${color_red}Invalid response, please try again:${color_norm}"
		build-kubemark-image
	fi
}

# Clean up repository
function clean-repo {
	echo -n -e "Do you want to remove build output and binary? [y/N]${color_cyan}>${color_norm} "
	read -r ISCLEAN
	if [[ "${ISCLEAN}" = "y" ]]; then
		echo -e "${color_yellow}CLEANING REPO${color_norm}"
		rm -rf "${KUBE_ROOT}/_output"
		rm -f "${KUBEMARK_IMAGE_LOCATION}/kubemark"
	elif [[ "${ISCLEAN}" = "N" ]]; then
		echo -n ""
	else
		echo -e "${color_red}Invalid response, please try again:${color_norm}"
		clean-repo
	fi
}
