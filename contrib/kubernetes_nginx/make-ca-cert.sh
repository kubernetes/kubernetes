#!/bin/bash

# Copyright 2014 Google Inc. All rights reserved.
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

cert_ip=$1

# TODO: Add support for discovery on other providers?
if [ "$cert_ip" == "_use_gce_external_ip_" ]; then
  cert_ip=$(curl -s -H Metadata-Flavor:Google http://metadata.google.internal./computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip)
fi  

tmpdir=$(mktemp -d --tmpdir kubernetes_cacert.XXXXXX)
trap 'rm -rf "${tmpdir}"' EXIT
cd "${tmpdir}"

# TODO: For now, this is a patched repo that makes subject-alt-name work, when the fix is upstream
#  move back to the upstream easyrsa
curl -L -J -O https://github.com/brendandburns/easy-rsa/archive/master.tar.gz > /dev/null 2>&1
tar xzf easy-rsa-master.tar.gz > /dev/null 2>&1

cd easy-rsa-master/easyrsa3
./easyrsa init-pki > /dev/null 2>&1
./easyrsa --batch build-ca nopass > /dev/null 2>&1
./easyrsa --subject-alt-name=IP:$cert_ip build-server-full kubernetes-master nopass > /dev/null 2>&1
./easyrsa build-client-full kubecfg nopass > /dev/null 2>&1
cp -p pki/issued/kubernetes-master.crt /usr/share/nginx/server.cert > /dev/null 2>&1
cp -p pki/private/kubernetes-master.key /usr/share/nginx/server.key > /dev/null 2>&1
cp -p pki/ca.crt /usr/share/nginx/ca.crt
cp -p pki/issued/kubecfg.crt /usr/share/nginx/kubecfg.crt
cp -p pki/private/kubecfg.key /usr/share/nginx/kubecfg.key