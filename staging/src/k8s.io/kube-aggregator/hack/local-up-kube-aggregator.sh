#!/usr/bin/env bash

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

set -o errexit
set -o nounset
set -o pipefail

AGG_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
KUBE_ROOT=${AGG_ROOT}/../../../..
source "${KUBE_ROOT}/hack/lib/init.sh"

AGGREGATOR_SECURE_PORT=${AGGREGATOR_SECURE_PORT:-31090}
API_HOST=${API_HOST:-localhost}
API_HOST_IP=${API_HOST_IP:-"127.0.0.1"}
AGGREGATOR_CERT_DIR=${AGGREGATOR_CERT_DIR:-"/var/run/kubernetes/aggregator"}

KUBE_CERT_DIR=${KUBE_CERT_DIR:-"/var/run/kubernetes"}
SERVING_CERT_CA_CERT=${SERVING_CERT_CA_CERT:-"${KUBE_CERT_DIR}/server-ca.crt"}
CLIENT_CERT_CA_CERT=${CLIENT_CERT_CA_CERT:-"${KUBE_CERT_DIR}/client-ca.crt"}
FRONT_PROXY_CLIENT_CERT_CA_CERT=${FRONT_PROXY_CLIENT_CERT_CA_CERT:-"${KUBE_CERT_DIR}/request-header-ca.crt"}
SERVING_CERT=${SERVING_CERT:-"${KUBE_CERT_DIR}/serving-kube-aggregator.crt"}
SERVING_KEY=${SERVING_KEY:-"${KUBE_CERT_DIR}/serving-kube-aggregator.key"}
FRONT_PROXY_CLIENT_CERT=${FRONT_PROXY_CLIENT_CERT:-"${KUBE_CERT_DIR}/client-auth-proxy.crt"}
FRONT_PROXY_CLIENT_KEY=${FRONT_PROXY_CLIENT_KEY:-"${KUBE_CERT_DIR}/client-auth-proxy.key"}


# Ensure AGGREGATOR_CERT_DIR is created for auto-generated crt/key and kubeconfig
mkdir -p "${AGGREGATOR_CERT_DIR}" &>/dev/null || sudo mkdir -p "${AGGREGATOR_CERT_DIR}"
sudo=$(test -w "${AGGREGATOR_CERT_DIR}" || echo "sudo -E")

# start_kube-aggregator relies on certificates created by start_apiserver
function start_kube-aggregator {
	 # Create serving and client CA.  etcd only takes one arg
	kube::util::create_signing_certkey "${sudo}" "${AGGREGATOR_CERT_DIR}" "etcd" '"client auth","server auth"'
	kube::util::create_serving_certkey "${sudo}" "${AGGREGATOR_CERT_DIR}" "etcd-ca" etcd etcd.kube-public.svc
	# etcd doesn't seem to have separate signers for serving and client trust
	kube::util::create_client_certkey "${sudo}" "${AGGREGATOR_CERT_DIR}" "etcd-ca" kube-aggregator-etcd kube-aggregator-etcd

	# don't fail if the namespace already exists or something
	# If this fails for some reason, the script will fail during creation of other resources
	kubectl create namespace kube-public || true

	# grant permission to run delegated authentication and authorization checks
	kubectl delete clusterrolebinding kube-aggregator:system:auth-delegator > /dev/null 2>&1 || true
	kubectl delete clusterrolebinding kube-aggregator:system:kube-aggregator > /dev/null 2>&1 || true
	kubectl create clusterrolebinding kube-aggregator:system:auth-delegator --clusterrole=system:auth-delegator --serviceaccount=kube-public:kube-aggregator
	kubectl create clusterrolebinding kube-aggregator:system:kube-aggregator --clusterrole=system:kube-aggregator --serviceaccount=kube-public:kube-aggregator
	kubectl delete rolebinding -n kube-system kube-aggregator:authentication-reader > /dev/null 2>&1 || true
	kubectl create rolebinding -n kube-system kube-aggregator:authentication-reader --role=extension-apiserver-authentication-reader --serviceaccount=kube-public:kube-aggregator

	# make sure the resources we're about to create don't exist
	kubectl -n kube-public delete secret auth-proxy-client serving-etcd serving-kube-aggregator kube-aggregator-etcd > /dev/null 2>&1 || true
	kubectl -n kube-public delete configmap etcd-ca kube-aggregator-ca client-ca request-header-ca > /dev/null 2>&1 || true
	kubectl -n kube-public delete -f "${AGG_ROOT}/artifacts/self-contained" > /dev/null 2>&1 || true

	${sudo} "$(which kubectl)" -n kube-public create secret tls auth-proxy-client --cert="${FRONT_PROXY_CLIENT_CERT}" --key="${FRONT_PROXY_CLIENT_KEY}"
	${sudo} "$(which kubectl)"  -n kube-public create secret tls serving-etcd --cert="${AGGREGATOR_CERT_DIR}/serving-etcd.crt" --key="${AGGREGATOR_CERT_DIR}/serving-etcd.key"
	${sudo} "$(which kubectl)"  -n kube-public create secret tls serving-kube-aggregator --cert="${SERVING_CERT}" --key="${SERVING_KEY}"
	${sudo} "$(which kubectl)"  -n kube-public create secret tls kube-aggregator-etcd --cert="${AGGREGATOR_CERT_DIR}/client-kube-aggregator-etcd.crt" --key="${AGGREGATOR_CERT_DIR}/client-kube-aggregator-etcd.key"
	kubectl -n kube-public create configmap etcd-ca --from-file="ca.crt=${AGGREGATOR_CERT_DIR}/etcd-ca.crt" || true
	kubectl -n kube-public create configmap kube-aggregator-ca --from-file="ca.crt=${SERVING_CERT_CA_CERT}" || true
	kubectl -n kube-public create configmap client-ca --from-file="ca.crt=${CLIENT_CERT_CA_CERT}" || true
	kubectl -n kube-public create configmap request-header-ca --from-file="ca.crt=${FRONT_PROXY_CLIENT_CERT_CA_CERT}" || true

	kubectl -n kube-public create -f "${AGG_ROOT}/artifacts/self-contained"

	# Wait for kube-aggregator to come up before launching the rest of the components.
	# This should work since we're creating a node port service.
	echo "Waiting for kube-aggregator to come up: https://${API_HOST_IP}:${AGGREGATOR_SECURE_PORT}/version"
	kube::util::wait_for_url "https://${API_HOST_IP}:${AGGREGATOR_SECURE_PORT}/version" "kube-aggregator: " 1 60 || exit 1
}

kube::util::test_openssl_installed
kube::util::ensure-cfssl

start_kube-aggregator

echo "kube-aggregator available at https://${API_HOST_IP}:${AGGREGATOR_SECURE_PORT} from 'api.kube-public.svc'"
