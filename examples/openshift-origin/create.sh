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

# Creates resources from the example, assumed to be run from Kubernetes repo root
export OPENSHIFT_EXAMPLE=$(pwd)/examples/openshift-origin
export OPENSHIFT_CONFIG=${OPENSHIFT_EXAMPLE}/config
mkdir ${OPENSHIFT_CONFIG}
cluster/kubectl.sh config view --output=yaml --flatten=true --minify=true > ${OPENSHIFT_CONFIG}/kubeconfig
cluster/kubectl.sh create -f $OPENSHIFT_EXAMPLE/openshift-service.yaml
sleep 30
export PUBLIC_IP=$(cluster/kubectl.sh get services openshift --template="{{ index .spec.publicIPs 0 }}")
echo $PUBLIC_IP
export PORTAL_IP=$(cluster/kubectl.sh get services openshift --template="{{ .spec.portalIP }}")
echo $PORTAL_IP
docker run --privileged -v ${OPENSHIFT_CONFIG}:/config openshift/origin start master --write-config=/config --kubeconfig=/config/kubeconfig --master=https://localhost:8443 --public-master=https://${PUBLIC_IP}:8443
sudo -E chown ${USER} -R ${OPENSHIFT_CONFIG}
docker run -i -t --privileged -e="OPENSHIFTCONFIG=/config/admin.kubeconfig" -v ${OPENSHIFT_CONFIG}:/config openshift/origin ex bundle-secret openshift-config -f /config &> ${OPENSHIFT_EXAMPLE}/secret.json
cluster/kubectl.sh create -f ${OPENSHIFT_EXAMPLE}/secret.json
cluster/kubectl.sh create -f ${OPENSHIFT_EXAMPLE}/openshift-controller.yaml
cluster/kubectl.sh get pods | grep openshift