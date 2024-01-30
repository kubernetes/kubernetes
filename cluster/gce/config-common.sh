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

# Returns the total number of Linux and Windows nodes in the cluster.
#
# Vars assumed:
#   NUM_NODES
#   NUM_WINDOWS_NODES
function get-num-nodes {
  echo "$((NUM_NODES + NUM_WINDOWS_NODES))"
}

# Vars assumed:
#   NUM_NODES
#   NUM_WINDOWS_NODES
function get-master-size {
  local suggested_master_size=2
  if [[ "$(get-num-nodes)" -gt "10" ]]; then
    suggested_master_size=4
  fi
  if [[ "$(get-num-nodes)" -gt "50" ]]; then
    suggested_master_size=8
  fi
  if [[ "$(get-num-nodes)" -gt "100" ]]; then
    suggested_master_size=16
  fi
  if [[ "$(get-num-nodes)" -gt "500" ]]; then
    suggested_master_size=32
  fi
  echo "${suggested_master_size}"
}

# Vars assumed:
#   NUM_NODES
#   NUM_WINDOWS_NODES
function get-master-root-disk-size() {
  local suggested_master_root_disk_size="20GB"
  if [[ "$(get-num-nodes)" -gt "500" ]]; then
    suggested_master_root_disk_size="100GB"
  fi
  if [[ "$(get-num-nodes)" -gt "3000" ]]; then
    suggested_master_root_disk_size="500GB"
  fi
  echo "${suggested_master_root_disk_size}"
}

# Vars assumed:
#   NUM_NODES
#   NUM_WINDOWS_NODES
function get-master-disk-size() {
  local suggested_master_disk_size="20GB"
  if [[ "$(get-num-nodes)" -gt "500" ]]; then
    suggested_master_disk_size="100GB"
  fi
  if [[ "$(get-num-nodes)" -gt "3000" ]]; then
    suggested_master_disk_size="200GB"
  fi
  echo "${suggested_master_disk_size}"
}

function get-node-ip-range {
  if [[ -n "${NODE_IP_RANGE:-}" ]]; then
    echo "${NODE_IP_RANGE}"
    return
  fi
  local suggested_range="10.40.0.0/22"
  if [[ "$(get-num-nodes)" -gt 1000 ]]; then
    suggested_range="10.40.0.0/21"
  fi
  if [[ "$(get-num-nodes)" -gt 2000 ]]; then
    suggested_range="10.40.0.0/20"
  fi
  if [[ "$(get-num-nodes)" -gt 4000 ]]; then
    suggested_range="10.40.0.0/19"
  fi
  echo "${suggested_range}"
}

function get-cluster-ip-range {
  local suggested_range="10.64.0.0/14"
  if [[ "$(get-num-nodes)" -gt 1000 ]]; then
    suggested_range="10.64.0.0/13"
  fi
  if [[ "$(get-num-nodes)" -gt 2000 ]]; then
    suggested_range="10.64.0.0/12"
  fi
  if [[ "$(get-num-nodes)" -gt 4000 ]]; then
    suggested_range="10.64.0.0/11"
  fi
  echo "${suggested_range}"
}

# Calculate ip alias range based on max number of pods.
# Let pow be the smallest integer which is bigger or equal to log2($1 * 2).
# (32 - pow) will be returned.
#
# $1: The number of max pods limitation.
function get-alias-range-size() {
  for pow in {0..31}; do
    if (( 1 << pow >= $1 * 2 )); then
      echo $((32 - pow))
      return 0
    fi
  done
}
# NOTE: Avoid giving nodes empty scopes, because kubelet needs a service account
# in order to initialize properly.
NODE_SCOPES="${NODE_SCOPES:-monitoring,logging-write,storage-ro}"

# Below exported vars are used in cluster/gce/util.sh (or maybe somewhere else),
# please remove those vars when not needed any more.

# Root directory for Kubernetes files on Windows nodes.
WINDOWS_K8S_DIR="C:\etc\kubernetes"
# Directory where Kubernetes binaries will be installed on Windows nodes.
export WINDOWS_NODE_DIR="${WINDOWS_K8S_DIR}\node\bin"
# Directory where Kubernetes log files will be stored on Windows nodes.
export WINDOWS_LOGS_DIR="${WINDOWS_K8S_DIR}\logs"
# Directory where CNI binaries will be stored on Windows nodes.
export WINDOWS_CNI_DIR="${WINDOWS_K8S_DIR}\cni"
# Directory where CNI config files will be stored on Windows nodes.
export WINDOWS_CNI_CONFIG_DIR="${WINDOWS_K8S_DIR}\cni\config"
# CNI storage path for Windows nodes
export WINDOWS_CNI_STORAGE_PATH="https://storage.googleapis.com/k8s-artifacts-cni/release"
# CNI version for Windows nodes
export WINDOWS_CNI_VERSION="v1.4.0"
# Pod manifests directory for Windows nodes on Windows nodes.
export WINDOWS_MANIFESTS_DIR="${WINDOWS_K8S_DIR}\manifests"
# Directory where cert/key files will be stores on Windows nodes.
export WINDOWS_PKI_DIR="${WINDOWS_K8S_DIR}\pki"
# Location of the certificates file on Windows nodes.
export WINDOWS_CA_FILE="${WINDOWS_PKI_DIR}\ca-certificates.crt"
# Path for kubelet config file on Windows nodes.
export WINDOWS_KUBELET_CONFIG_FILE="${WINDOWS_K8S_DIR}\kubelet-config.yaml"
# Path for kubeconfig file on Windows nodes.
export WINDOWS_KUBECONFIG_FILE="${WINDOWS_K8S_DIR}\kubelet.kubeconfig"
# Path for bootstrap kubeconfig file on Windows nodes.
export WINDOWS_BOOTSTRAP_KUBECONFIG_FILE="${WINDOWS_K8S_DIR}\kubelet.bootstrap-kubeconfig"
# Path for kube-proxy kubeconfig file on Windows nodes.
export WINDOWS_KUBEPROXY_KUBECONFIG_FILE="${WINDOWS_K8S_DIR}\kubeproxy.kubeconfig"
# Path for kube-proxy kubeconfig file on Windows nodes.
export WINDOWS_NODEPROBLEMDETECTOR_KUBECONFIG_FILE="${WINDOWS_K8S_DIR}\node-problem-detector.kubeconfig"
# Pause container image for Windows container.
export WINDOWS_INFRA_CONTAINER="registry.k8s.io/pause:3.9"
# Storage Path for csi-proxy. csi-proxy only needs to be installed for Windows.
export CSI_PROXY_STORAGE_PATH="https://storage.googleapis.com/gke-release/csi-proxy"
# Version for csi-proxy
export CSI_PROXY_VERSION="${CSI_PROXY_VERSION:-v1.1.1-gke.0}"
# csi-proxy additional flags, there are additional flags that cannot be unset in k8s-node-setup.psm1
export CSI_PROXY_FLAGS="${CSI_PROXY_FLAGS:-}"
# Storage path for auth-provider-gcp binaries
export AUTH_PROVIDER_GCP_STORAGE_PATH="${AUTH_PROVIDER_GCP_STORAGE_PATH:-https://storage.googleapis.com/gke-release/auth-provider-gcp}"
# auth-provider-gcp version
export AUTH_PROVIDER_GCP_VERSION="${AUTH_PROVIDER_GCP_VERSION:-v0.0.2-gke.4}"
# Hash of auth-provider-gcp.exe binary
export AUTH_PROVIDER_GCP_HASH_WINDOWS_AMD64="${AUTH_PROVIDER_GCP_HASH_WINDOWS_AMD64:-348af2c189d938e1a4fa5ac5c640d21e003da1f000abcd6fd7eef2acd0678638286e40703618758d4fdfe2cc4b90e920f0422128ec777c74054af9dd4405de12}"
# Directory of kubelet image credential provider binary files on windows
export AUTH_PROVIDER_GCP_LINUX_BIN_DIR="${AUTH_PROVIDER_GCP_LINUX_BIN_DIR:-/home/kubernetes/bin}"
# Location of kubelet image credential provider config file on windows
export AUTH_PROVIDER_GCP_LINUX_CONF_FILE="${AUTH_PROVIDER_GCP_LINUX_CONF_FILE:-/home/kubernetes/cri-auth-config.yaml}"
# Directory of kubelet image credential provider binary files on windows
export AUTH_PROVIDER_GCP_WINDOWS_BIN_DIR=${AUTH_PROVIDER_GCP_WINDOWS_BIN_DIR:-${WINDOWS_NODE_DIR}}
# Location of kubelet image credential provider config file on windows
export AUTH_PROVIDER_GCP_WINDOWS_CONF_FILE="${AUTH_PROVIDER_GCP_WINDOWS_CONF_FILE:-${WINDOWS_K8S_DIR}\cri-auth-config.yaml}"
