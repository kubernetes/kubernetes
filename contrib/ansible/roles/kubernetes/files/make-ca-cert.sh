#!/bin/bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
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

set -o errexit
set -o nounset
set -o pipefail

# Caller should set in the ev:
# MASTER_IP - this may be an ip or things like "_use_gce_external_ip_"
# MASTER_NAME - DNS name for the master
# DNS_DOMAIN - which will be passed to minions in --cluster-domain
# SERVICE_CLUSTER_IP_RANGE - where all service IPs are allocated

# Also the following will be respected
# CERT_DIR - where to place the finished certs
# CERT_GROUP - who the group owner of the cert files should be

cert_ip="${MASTER_IP:="${1}"}"
master_name="${MASTER_NAME:="kubernetes"}"
service_range="${SERVICE_CLUSTER_IP_RANGE:="10.0.0.0/16"}"
dns_domain="${DNS_DOMAIN:="cluster.local"}"
cert_dir="${CERT_DIR:-"/srv/kubernetes"}"
cert_group="${CERT_GROUP:="kube-cert"}"

# The following certificate pairs are created:
#
#  - ca (the cluster's certificate authority)
#  - server
#  - kubelet
#  - kubecfg (for kubectl)
#
# TODO(roberthbailey): Replace easyrsa with a simple Go program to generate
# the certs that we need.

# TODO: Add support for discovery on other providers?
if [ "$cert_ip" == "_use_gce_external_ip_" ]; then
  cert_ip=$(curl -s -H Metadata-Flavor:Google http://metadata.google.internal./computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip)
fi

if [ "$cert_ip" == "_use_aws_external_ip_" ]; then
  cert_ip=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
fi

if [ "$cert_ip" == "_use_azure_dns_name_" ]; then
  cert_ip=$(uname -n | awk -F. '{ print $2 }').cloudapp.net
fi

tmpdir=$(mktemp -d --tmpdir kubernetes_cacert.XXXXXX)
trap 'rm -rf "${tmpdir}"' EXIT
cd "${tmpdir}"

# TODO: For now, this is a patched tool that makes subject-alt-name work, when
# the fix is upstream  move back to the upstream easyrsa.  This is cached in GCS
# but is originally taken from:
#   https://github.com/brendandburns/easy-rsa/archive/master.tar.gz
#
# To update, do the following:
# curl -o easy-rsa.tar.gz https://github.com/brendandburns/easy-rsa/archive/master.tar.gz
# gsutil cp easy-rsa.tar.gz gs://kubernetes-release/easy-rsa/easy-rsa.tar.gz
# gsutil acl ch -R -g all:R gs://kubernetes-release/easy-rsa/easy-rsa.tar.gz
#
# Due to GCS caching of public objects, it may take time for this to be widely
# distributed.

# Calculate the first ip address in the service range
octets=($(echo "${service_range}" | sed -e 's|/.*||' -e 's/\./ /g'))
((octets[3]+=1))
service_ip=$(echo "${octets[*]}" | sed 's/ /./g')

# Determine appropriete subject alt names
sans="IP:${cert_ip},IP:${service_ip},DNS:kubernetes,DNS:kubernetes.default,DNS:kubernetes.default.svc,DNS:kubernetes.default.svc.${dns_domain},DNS:${master_name}"

curl -L -O https://storage.googleapis.com/kubernetes-release/easy-rsa/easy-rsa.tar.gz > /dev/null 2>&1
tar xzf easy-rsa.tar.gz > /dev/null
cd easy-rsa-master/easyrsa3

(./easyrsa init-pki > /dev/null 2>&1
 ./easyrsa --batch "--req-cn=${cert_ip}@$(date +%s)" build-ca nopass > /dev/null 2>&1
 ./easyrsa --subject-alt-name="${sans}" build-server-full "${master_name}" nopass > /dev/null 2>&1
 ./easyrsa build-client-full kubelet nopass > /dev/null 2>&1
 ./easyrsa build-client-full kubecfg nopass > /dev/null 2>&1) || {
 # If there was an error in the subshell, just die.
 # TODO(roberthbailey): add better error handling here
 echo "=== Failed to generate certificates: Aborting ==="
 exit 2
 }

mkdir -p "$cert_dir"

cp -p pki/ca.crt "${cert_dir}/ca.crt"
cp -p "pki/issued/${master_name}.crt" "${cert_dir}/server.crt" > /dev/null 2>&1
cp -p "pki/private/${master_name}.key" "${cert_dir}/server.key" > /dev/null 2>&1
cp -p pki/issued/kubecfg.crt "${cert_dir}/kubecfg.crt"
cp -p pki/private/kubecfg.key "${cert_dir}/kubecfg.key"
cp -p pki/issued/kubelet.crt "${cert_dir}/kubelet.crt"
cp -p pki/private/kubelet.key "${cert_dir}/kubelet.key"

CERTS=("ca.crt" "server.key" "server.crt" "kubelet.key" "kubelet.crt" "kubecfg.key" "kubecfg.crt")
for cert in "${CERTS[@]}"; do
  chgrp "${cert_group}" "${cert_dir}/${cert}"
  chmod 660 "${cert_dir}/${cert}"
done
