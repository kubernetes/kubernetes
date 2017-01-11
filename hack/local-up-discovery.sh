#!/bin/bash

# Copyright 2016 The Kubernetes Authors.
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

# starts kube-aggregator as a pod after you've run `local-up-cluster.sh`


KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

DISCOVERY_SECURE_PORT=${DISCOVERY_SECURE_PORT:-31090}
API_HOST=${API_HOST:-localhost}
API_HOST_IP=${API_HOST_IP:-"127.0.0.1"}
CERT_DIR=${CERT_DIR:-"/var/run/kubernetes"}
ROOT_CA_FILE=$CERT_DIR/apiserver.crt

# Ensure CERT_DIR is created for auto-generated crt/key and kubeconfig
mkdir -p "${CERT_DIR}" &>/dev/null || sudo mkdir -p "${CERT_DIR}"
sudo=$(test -w "${CERT_DIR}" || echo "sudo -E")


kubectl=$(kube::util::find-binary kubectl)

function kubectl_core {
	${kubectl} --kubeconfig="${CERT_DIR}/admin.kubeconfig" $@
}

function sudo_kubectl_core {
	${sudo} ${kubectl} --kubeconfig="${CERT_DIR}/admin.kubeconfig" $@
}

# start_discovery relies on certificates created by start_apiserver
function start_discovery {
	kube::util::create_signing_certkey "${sudo}" "${CERT_DIR}" "discovery" '"server auth"'
	# sign the discovery cert to be good for the local node too, so that we can trust it
	kube::util::create_serving_certkey "${sudo}" "${CERT_DIR}" "discovery-ca" discovery api.kube-public.svc "localhost" ${API_HOST_IP}

	 # Create serving and client CA.  etcd only takes one arg
	kube::util::create_signing_certkey "${sudo}" "${CERT_DIR}" "etcd" '"client auth","server auth"'
	kube::util::create_serving_certkey "${sudo}" "${CERT_DIR}" "etcd-ca" etcd etcd.kube-public.svc
	# etcd doesn't seem to have separate signers for serving and client trust
	kube::util::create_client_certkey "${sudo}" "${CERT_DIR}" "etcd-ca" discovery-etcd discovery-etcd

	# don't fail if the namespace already exists or something
	# If this fails for some reason, the script will fail during creation of other resources
	kubectl_core create namespace kube-public || true

	# grant permission to run delegated authentication and authorization checks
	kubectl_core delete clusterrolebinding discovery:system:auth-delegator > /dev/null 2>&1 || true
	kubectl_core delete clusterrolebinding discovery:system:kube-aggregator > /dev/null 2>&1 || true
	kubectl_core create clusterrolebinding discovery:system:auth-delegator --clusterrole=system:auth-delegator --serviceaccount=kube-public:kube-aggregator
	kubectl_core create clusterrolebinding discovery:system:kube-aggregator --clusterrole=system:kube-aggregator --serviceaccount=kube-public:kube-aggregator

	# make sure the resources we're about to create don't exist
	kubectl_core -n kube-public delete secret auth-proxy-client serving-etcd serving-discovery discovery-etcd > /dev/null 2>&1 || true
	kubectl_core -n kube-public delete configmap etcd-ca discovery-ca client-ca request-header-ca > /dev/null 2>&1 || true
	kubectl_core -n kube-public delete -f "${KUBE_ROOT}/cmd/kube-aggregator/artifacts/local-cluster-up" > /dev/null 2>&1 || true

	sudo_kubectl_core -n kube-public create secret tls auth-proxy-client --cert="${CERT_DIR}/client-auth-proxy.crt" --key="${CERT_DIR}/client-auth-proxy.key"
	sudo_kubectl_core -n kube-public create secret tls serving-etcd --cert="${CERT_DIR}/serving-etcd.crt" --key="${CERT_DIR}/serving-etcd.key"
	sudo_kubectl_core -n kube-public create secret tls serving-discovery --cert="${CERT_DIR}/serving-discovery.crt" --key="${CERT_DIR}/serving-discovery.key"
	sudo_kubectl_core -n kube-public create secret tls discovery-etcd --cert="${CERT_DIR}/client-discovery-etcd.crt" --key="${CERT_DIR}/client-discovery-etcd.key"
	kubectl_core -n kube-public create configmap etcd-ca --from-file="ca.crt=${CERT_DIR}/etcd-ca.crt" || true
	kubectl_core -n kube-public create configmap discovery-ca --from-file="ca.crt=${CERT_DIR}/discovery-ca.crt" || true
	kubectl_core -n kube-public create configmap client-ca --from-file="ca.crt=${CERT_DIR}/client-ca.crt" || true
	kubectl_core -n kube-public create configmap request-header-ca --from-file="ca.crt=${CERT_DIR}/request-header-ca.crt" || true

	${KUBE_ROOT}/cmd/kube-aggregator/hack/build-image.sh

	kubectl_core -n kube-public create -f "${KUBE_ROOT}/cmd/kube-aggregator/artifacts/local-cluster-up"

	${sudo} cp "${CERT_DIR}/admin.kubeconfig" "${CERT_DIR}/admin-discovery.kubeconfig"
	${sudo} chown ${USER} "${CERT_DIR}/admin-discovery.kubeconfig"
	${kubectl} config set-cluster local-up-cluster --kubeconfig="${CERT_DIR}/admin-discovery.kubeconfig" --certificate-authority="${CERT_DIR}/discovery-ca.crt" --embed-certs --server="https://${API_HOST_IP}:${DISCOVERY_SECURE_PORT}"

	# Wait for kube-aggregator to come up before launching the rest of the components.
	# This should work since we're creating a node port service.
	echo "Waiting for kube-aggregator to come up: https://${API_HOST_IP}:${DISCOVERY_SECURE_PORT}/version"
	kube::util::wait_for_url "https://${API_HOST_IP}:${DISCOVERY_SECURE_PORT}/version" "kube-aggregator: " 1 60 || exit 1

	# something is weird with the proxy
	sleep 1

	# create the "normal" api services for the core API server
	${kubectl} --kubeconfig="${CERT_DIR}/admin-discovery.kubeconfig" create -f "${KUBE_ROOT}/cmd/kube-aggregator/artifacts/core-apiservices"
}

kube::util::test_openssl_installed
kube::util::test_cfssl_installed

start_discovery

echo "kuberentes-discovery available at https://${API_HOST_IP}:${DISCOVERY_SECURE_PORT} from 'api.kube-public.svc'"
