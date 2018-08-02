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

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../../..

# Creates a new kube-spawn cluster
function create-clusters {
	echo -e "${color_yellow}CHECKING CLUSTERS${color_norm}"
	if bx cs clusters | grep -Fq 'deleting'; then
		echo -n "Deleting old clusters"
	fi
	while bx cs clusters | grep -Fq 'deleting'
	do
		echo -n "."
		sleep 10
	done
	echo ""
	bx cs region-set us-east >/dev/null
	bx cs vlans wdc06 >/dev/null
	PRIVLAN=$(bx cs vlans wdc06 --json | jq '. | .[] | select(.type == "private") | .id' | sed -e "s/\"//g")
	PUBVLAN=$(bx cs vlans wdc06 --json | jq '. | .[] | select(.type == "public") | .id' | sed -e "s/\"//g")
	if ! bx cs clusters | grep -Fq 'kubeSpawnTester'; then
		echo "Creating spawning cluster"
		bx cs cluster-create --location ${CLUSTER_LOCATION} --public-vlan ${PUBVLAN} --private-vlan ${PRIVLAN} --workers 2 --machine-type u2c.2x4 --name kubeSpawnTester
	fi
	if ! bx cs clusters | grep -Fq 'kubeMasterTester'; then
		echo "Creating master cluster"
		bx cs cluster-create --location ${CLUSTER_LOCATION} --public-vlan ${PUBVLAN} --private-vlan ${PRIVLAN} --workers 2 --machine-type u2c.2x4 --name kubeMasterTester
	fi
	push-image
	if ! bx cs clusters | grep 'kubeSpawnTester' | grep -Fq 'normal'; then
		echo -e "${color_cyan}Warning: new clusters may take up to 60 minutes to be ready${color_norm}"
		echo -n "Clusters loading"
	fi
	while ! bx cs clusters | grep 'kubeSpawnTester' | grep -Fq 'normal' 
	do
		echo -n "."
		sleep 5
	done
	while ! bx cs clusters | grep 'kubeMasterTester' | grep -Fq 'normal'
	do
		echo -n "."
		sleep 5
	done
	echo -e "${color_yellow}CLUSTER CREATION COMPLETE${color_norm}"
}

# Builds and pushes image to registry
function push-image {
	if [[ "${ISBUILD}" = "y" ]]; then
		if ! bx cr namespaces | grep -Fq ${KUBE_NAMESPACE}; then
			echo "Creating registry namespace"
			bx cr namespace-add ${KUBE_NAMESPACE}
			echo "bx cr namespace-rm ${KUBE_NAMESPACE}" >> ${RESOURCE_DIRECTORY}/iks-namespacelist.sh
		fi
		docker build -t ${KUBEMARK_INIT_TAG} ${KUBEMARK_IMAGE_LOCATION}
		docker tag ${KUBEMARK_INIT_TAG} ${KUBEMARK_IMAGE_REGISTRY}${KUBE_NAMESPACE}/${PROJECT}:${KUBEMARK_IMAGE_TAG}
		docker push ${KUBEMARK_IMAGE_REGISTRY}${KUBE_NAMESPACE}/${PROJECT}:${KUBEMARK_IMAGE_TAG}
		echo "Image pushed"
	else
		KUBEMARK_IMAGE_REGISTRY=$(echo "brandondr96")
		KUBE_NAMESPACE=""
	fi
}

# Allow user to use existing clusters if desired
function choose-clusters {
	echo -n -e "Do you want to use custom clusters? [y/N]${color_cyan}>${color_norm} "
	read USE_EXISTING
	if [[ "${USE_EXISTING}" = "y" ]]; then
		echo -e "${color_yellow}Enter path for desired hollow-node spawning cluster kubeconfig file:${color_norm}"
		read CUSTOM_SPAWN_CONFIG
		echo -e "${color_yellow}Enter path for desired hollow-node hosting cluster kubeconfig file:${color_norm}"
		read CUSTOM_MASTER_CONFIG
		push-image
	elif [[ "${USE_EXISTING}" = "N" ]]; then
		create-clusters
	else
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

# Sets hollow nodes spawned under master
function set-hollow-master {
	echo -e "${color_yellow}CONFIGURING MASTER${color_norm}"
	master-config
	MASTER_IP=$(cat $KUBECONFIG | grep server | awk -F "/" '{print $3}')
}

# Set up master cluster environment
function master-config {
	if [[ "${USE_EXISTING}" = "y" ]]; then
		export KUBECONFIG=${CUSTOM_MASTER_CONFIG}
	else
		$(bx cs cluster-config kubeMasterTester --admin | grep export)
	fi
}

# Set up spawn cluster environment
function spawn-config {
	if [[ "${USE_EXISTING}" = "y" ]]; then
		export KUBECONFIG=${CUSTOM_SPAWN_CONFIG}
	else
		$(bx cs cluster-config kubeSpawnTester --admin | grep export)
	fi
}

# Deletes existing clusters
function delete-clusters {
	echo "DELETING CLUSTERS"
 	bx cs cluster-rm kubeSpawnTester
 	bx cs cluster-rm kubeMasterTester
 	while ! bx cs clusters | grep 'kubeSpawnTester' | grep -Fq 'deleting'
 	do
 		sleep 5
 	done
	while ! bx cs clusters | grep 'kubeMasterTester' | grep -Fq 'deleting'
	do
		sleep 5
	done
	kubectl delete ns kubemark
}

# Login to cloud services
function complete-login {
	echo -e "${color_yellow}LOGGING INTO CLOUD SERVICES${color_norm}"
	echo -n -e "Do you have a federated IBM cloud login? [y/N]${color_cyan}>${color_norm} "
	read ISFED
	if [[ "${ISFED}" = "y" ]]; then
		bx login --sso -a ${REGISTRY_LOGIN_URL}
	elif [[ "${ISFED}" = "N" ]]; then
		bx login -a ${REGISTRY_LOGIN_URL}
	else
		echo -e "${color_red}Invalid response, please try again:${color_norm}"
		complete-login
	fi
	bx cr login
}

# Generate values to fill the hollow-node configuration
function generate-values {
	echo "Generating values"
	master-config
	KUBECTL=kubectl
	KUBEMARK_DIRECTORY="${KUBE_ROOT}/test/kubemark"
	RESOURCE_DIRECTORY="${KUBEMARK_DIRECTORY}/resources"
	TEST_CLUSTER_API_CONTENT_TYPE="bluemix" #Determine correct usage of this
	CONFIGPATH=${KUBECONFIG%/*}
	KUBELET_CERT_BASE64="${KUBELET_CERT_BASE64:-$(cat ${CONFIGPATH}/admin.pem | base64 | tr -d '\r\n')}"
	KUBELET_KEY_BASE64="${KUBELET_KEY_BASE64:-$(cat ${CONFIGPATH}/admin-key.pem  | base64 | tr -d '\r\n')}"
	CA_CERT_BASE64="${CA_CERT_BASE64:-$(cat `find ${CONFIGPATH} -name *ca*` | base64 | tr -d '\r\n')}"
}

# Build image for kubemark
function build-kubemark-image {
	echo -n -e "Do you want to build the kubemark image? [y/N]${color_cyan}>${color_norm} "
	read ISBUILD
	if [[ "${ISBUILD}" = "y" ]]; then
		echo -e "${color_yellow}BUILDING IMAGE${color_norm}"
		${KUBE_ROOT}/build/run.sh make kubemark
		cp ${KUBE_ROOT}/_output/dockerized/bin/linux/amd64/kubemark ${KUBEMARK_IMAGE_LOCATION}
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
	read ISCLEAN
	if [[ "${ISCLEAN}" = "y" ]]; then
		echo -e "${color_yellow}CLEANING REPO${color_norm}"
		rm -rf ${KUBE_ROOT}/_output
		rm -f ${KUBEMARK_IMAGE_LOCATION}/kubemark
	elif [[ "${ISCLEAN}" = "N" ]]; then
		echo -n ""
	else
		echo -e "${color_red}Invalid response, please try again:${color_norm}"
		clean-repo
	fi
}
