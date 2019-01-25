#!/usr/bin/env bash

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

set -o errexit
set -o nounset
set -o pipefail

DEBUG="${DEBUG:-false}"

if [ "${DEBUG}" == "true" ]; then
	set -x
fi

cert_ip=$1
extra_sans=${2:-}
cert_dir=${CERT_DIR:-/srv/kubernetes}
cert_group=${CERT_GROUP:-kube-cert}

mkdir -p "$cert_dir"

use_cn=false

sans="IP:${cert_ip}"
if [[ -n "${extra_sans}" ]]; then
  sans="${sans},${extra_sans}"
fi

tmpdir=$(mktemp -d -t kubernetes_cacert.XXXXXX)
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
#
# Use ~/kube/easy-rsa.tar.gz if it exists, so that it can be
# pre-pushed in cases where an outgoing connection is not allowed.
if [ -f ~/kube/easy-rsa.tar.gz ]; then
	ln -s ~/kube/easy-rsa.tar.gz .
else
	curl -L -O https://storage.googleapis.com/kubernetes-release/easy-rsa/easy-rsa.tar.gz > /dev/null 2>&1
fi
tar xzf easy-rsa.tar.gz > /dev/null 2>&1

cd easy-rsa-master/easyrsa3
./easyrsa init-pki > /dev/null 2>&1
./easyrsa --batch "--req-cn=${cert_ip}@$(date +%s)" build-ca nopass > /dev/null 2>&1
if [ $use_cn = "true" ]; then
    ./easyrsa build-server-full "${cert_ip}" nopass > /dev/null 2>&1
    cp -p "pki/issued/${cert_ip}.crt" "${cert_dir}/server.cert" > /dev/null 2>&1
    cp -p "pki/private/${cert_ip}.key" "${cert_dir}/server.key" > /dev/null 2>&1
else
    ./easyrsa --subject-alt-name="${sans}" build-server-full kubernetes-master nopass > /dev/null 2>&1
    cp -p pki/issued/kubernetes-master.crt "${cert_dir}/server.cert" > /dev/null 2>&1
    cp -p pki/private/kubernetes-master.key "${cert_dir}/server.key" > /dev/null 2>&1
fi
# Make a superuser client cert with subject "O=system:masters, CN=kubecfg"
./easyrsa --dn-mode=org \
  --req-cn=kubecfg --req-org=system:masters \
  --req-c= --req-st= --req-city= --req-email= --req-ou= \
  build-client-full kubecfg nopass > /dev/null 2>&1
cp -p pki/ca.crt "${cert_dir}/ca.crt"
cp -p pki/issued/kubecfg.crt "${cert_dir}/kubecfg.crt"
cp -p pki/private/kubecfg.key "${cert_dir}/kubecfg.key"
# Make server certs accessible to apiserver.
chgrp "${cert_group}" "${cert_dir}/server.key" "${cert_dir}/server.cert" "${cert_dir}/ca.crt"
chmod 660 "${cert_dir}/server.key" "${cert_dir}/server.cert" "${cert_dir}/ca.crt"
