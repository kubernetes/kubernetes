#!/bin/bash

# Copyright 2015 The Kubernetes Authors All rights reserved.
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

function configure-hostname() {
  hostnamectl set-hostname $(hostname | cut -f1 -d.)
}

function configure-kubelet() {
  mkdir -p /var/lib/kubelet
  cat > /var/lib/kubelet/kubeconfig << EOF
apiVersion: v1
kind: Config
users:
- name: kubelet
  user:
    client-certificate-data: ${KUBELET_CERT}
    client-key-data: ${KUBELET_KEY}
clusters:
- name: local
  cluster:
    certificate-authority-data: ${CA_CERT}
contexts:
- context:
    cluster: local
    user: kubelet
  name: service-account-context
current-context: service-account-context
EOF
}

function configure-kube-proxy() {
  mkdir -p /var/lib/kube-proxy
  cat > /var/lib/kube-proxy/kubeconfig << EOF
apiVersion: v1
kind: Config
users:
- name: kube-proxy
  user:
    token: ${KUBE_PROXY_TOKEN}
clusters:
- name: local
  cluster:
    certificate-authority-data: ${CA_CERT}
contexts:
- context:
    cluster: local
    user: kube-proxy
  name: service-account-context
current-context: service-account-context
EOF
}

function configure-static-pods() {
  mkdir -p /etc/kubernetes/manifests

  if [[ "${ENABLE_NODE_LOGGING}" == "true" ]];then
    if [[ "${LOGGING_DESTINATION}" == "gcp" ]];then
      echo "Placing fluentd-gcp"
      # fluentd-gcp
      cat > /etc/kubernetes/manifests/fluentd-gcp.yaml <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: fluentd-cloud-logging
  namespace: kube-system
spec:
  containers:
  - name: fluentd-cloud-logging
    image: gcr.io/google_containers/fluentd-gcp:1.14
    resources:
      limits:
        cpu: 100m
        memory: 200Mi
    env:
    - name: FLUENTD_ARGS
      value: -q
    volumeMounts:
    - name: varlog
      mountPath: /var/log
    - name: varlibdockercontainers
      mountPath: /var/lib/docker/containers
      readOnly: true
  terminationGracePeriodSeconds: 30
  volumes:
  - name: varlog
    hostPath:
      path: /var/log
  - name: varlibdockercontainers
    hostPath:
      path: /var/lib/docker/containers
EOF
    elif [[ "${LOGGING_DESTINATION}" == "elasticsearch" ]];then
      echo "Placing fluentd-es"
      # fluentd-es
      cat > /etc/kubernetes/manifests/fluentd-es.yaml <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: fluentd-elasticsearch
  namespace: kube-system
spec:
  containers:
  - name: fluentd-elasticsearch
    image: beta.gcr.io/google_containers/fluentd-elasticsearch:1.12
    resources:
      limits:
        cpu: 100m
    args:
    - -q
    volumeMounts:
    - name: varlog
      mountPath: /var/log
    - name: varlibdockercontainers
      mountPath: /var/lib/docker/containers
      readOnly: true
  terminationGracePeriodSeconds: 30
  volumes:
  - name: varlog
    hostPath:
      path: /var/log
  - name: varlibdockercontainers
    hostPath:
      path: /var/lib/docker/containers
EOF
    fi
  fi
}

####################################################################################

echo "Configuring hostname"
configure-hostname

echo "Configuring kubelet"
configure-kubelet

echo "Configuring kube-proxy"
configure-kube-proxy

echo "Configuring static pods"
configure-static-pods

echo "Finish configuration successfully!"
