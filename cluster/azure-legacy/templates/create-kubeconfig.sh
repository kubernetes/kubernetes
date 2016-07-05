#!/bin/bash

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

# Create the kube config file for kubelet and kube-proxy in minions.
# password and username required

function create-salt-kubelet-auth() {
  local -r kubelet_kubeconfig_file="/srv/salt-overlay/salt/kubelet/kubeconfig"
  mkdir -p /srv/salt-overlay/salt/kubelet
  (umask 077;
    cat > "${kubelet_kubeconfig_file}" <<EOF
apiVersion: v1
clusters:
- cluster:
    insecure-skip-tls-verify: true
    server: https://${KUBE_MASTER_IP}
  name: azure_kubernetes
contexts:
- context:
    cluster: azure_kubernetes
    user: kubelet
  name: azure_kubernetes
current-context: azure_kubernetes
kind: Config
preferences: {}
users:
- name: kubelet
  user:
    password: ${KUBE_PASSWORD}
    username: ${KUBE_USER}
EOF
)
}

function create-salt-kube-proxy-auth() {
  local -r kube_proxy_kubeconfig_file="/srv/salt-overlay/salt/kube-proxy/kubeconfig"
  mkdir -p /srv/salt-overlay/salt/kube-proxy
  (umask 077;
    cat > "${kubelet_kubeconfig_file}" <<EOF
apiVersion: v1
clusters:
- cluster:
    insecure-skip-tls-verify: true
    server: https://${KUBE_MASTER_IP}
  name: azure_kubernetes
contexts:
- context:
    cluster: azure_kubernetes
    user: kube-proxy
  name: azure_kubernetes
current-context: azure_kubernetes
kind: Config
preferences: {}
users:
- name: kube-proxy
  user:
    password: ${KUBE_PASSWORD}
    username: ${KUBE_USER}
EOF
)
}

create-salt-kubelet-auth
create-salt-kube-proxy-auth
