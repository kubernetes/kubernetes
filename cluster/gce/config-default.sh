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

# TODO(jbeda): Provide a way to override project
# gcloud multiplexing for shared GCE/GKE tests.
KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/../..
source "${KUBE_ROOT}/cluster/gce/config-common.sh"

# Specifying KUBE_GCE_API_ENDPOINT will override the default GCE Compute API endpoint (https://www.googleapis.com/compute/v1/).
# This endpoint has to be pointing to v1 api. For example, https://www.googleapis.com/compute/staging_v1/
export GCE_API_ENDPOINT=${KUBE_GCE_API_ENDPOINT:-}
export GCLOUD=gcloud
ZONE=${KUBE_GCE_ZONE:-us-central1-b}
export REGION=${ZONE%-*}
RELEASE_REGION_FALLBACK=${RELEASE_REGION_FALLBACK:-false}
REGIONAL_KUBE_ADDONS=${REGIONAL_KUBE_ADDONS:-true}
NODE_SIZE=${NODE_SIZE:-e2-standard-2}
NUM_NODES=${NUM_NODES:-3}
NUM_WINDOWS_NODES=${NUM_WINDOWS_NODES:-0}
MASTER_SIZE=${MASTER_SIZE:-e2-standard-$(get-master-size)}
MASTER_MIN_CPU_ARCHITECTURE=${MASTER_MIN_CPU_ARCHITECTURE:-} # To allow choosing better architectures.
export MASTER_DISK_TYPE=pd-ssd
MASTER_DISK_SIZE=${MASTER_DISK_SIZE:-$(get-master-disk-size)}
MASTER_ROOT_DISK_SIZE=${MASTER_ROOT_DISK_SIZE:-$(get-master-root-disk-size)}
NODE_DISK_TYPE=${NODE_DISK_TYPE:-pd-standard}
NODE_DISK_SIZE=${NODE_DISK_SIZE:-100GB}
NODE_LOCAL_SSDS=${NODE_LOCAL_SSDS:-0}
NODE_LABELS="${KUBE_NODE_LABELS:-}"
WINDOWS_NODE_LABELS="${WINDOWS_NODE_LABELS:-}"
NODE_LOCAL_SSDS_EPHEMERAL=${NODE_LOCAL_SSDS_EPHEMERAL:-}

# KUBE_CREATE_NODES can be used to avoid creating nodes, while master will be sized for NUM_NODES nodes.
# Firewalls and node templates are still created.
KUBE_CREATE_NODES="${KUBE_CREATE_NODES:-true}"

# An extension to local SSDs allowing users to specify block/fs and SCSI/NVMe devices
# Format of this variable will be "#,scsi/nvme,block/fs" you can specify multiple
# configurations by separating them by a semi-colon ex. "2,scsi,fs;1,nvme,block"
# is a request for 2 SCSI formatted and mounted SSDs and 1 NVMe block device SSD.
NODE_LOCAL_SSDS_EXT=${NODE_LOCAL_SSDS_EXT:-}
# Accelerators to be attached to each node. Format "type=<accelerator-type>,count=<accelerator-count>"
# More information on available GPUs here - https://cloud.google.com/compute/docs/gpus/
NODE_ACCELERATORS=${NODE_ACCELERATORS:-""}
export REGISTER_MASTER_KUBELET=${REGISTER_MASTER:-true}
PREEMPTIBLE_NODE=${PREEMPTIBLE_NODE:-false}
PREEMPTIBLE_MASTER=${PREEMPTIBLE_MASTER:-false}
KUBE_DELETE_NODES=${KUBE_DELETE_NODES:-true}
KUBE_DELETE_NETWORK=${KUBE_DELETE_NETWORK:-} # default value calculated below
CREATE_CUSTOM_NETWORK=${CREATE_CUSTOM_NETWORK:-false}
MIG_WAIT_UNTIL_STABLE_TIMEOUT=${MIG_WAIT_UNTIL_STABLE_TIMEOUT:-1800}

MASTER_OS_DISTRIBUTION=${KUBE_MASTER_OS_DISTRIBUTION:-${KUBE_OS_DISTRIBUTION:-gci}}
NODE_OS_DISTRIBUTION=${KUBE_NODE_OS_DISTRIBUTION:-${KUBE_OS_DISTRIBUTION:-gci}}
WINDOWS_NODE_OS_DISTRIBUTION=${WINDOWS_NODE_OS_DISTRIBUTION:-win2019}

if [[ "${MASTER_OS_DISTRIBUTION}" == "cos" ]]; then
    MASTER_OS_DISTRIBUTION="gci"
fi

if [[ "${NODE_OS_DISTRIBUTION}" == "cos" ]]; then
    NODE_OS_DISTRIBUTION="gci"
fi

# GPUs supported in GCE do not have compatible drivers in Debian 7.
if [[ "${NODE_OS_DISTRIBUTION}" == "debian" ]]; then
    NODE_ACCELERATORS=""
fi

# By default a cluster will be started with the master and nodes
# on Container-optimized OS (cos, previously known as gci). If
# you are updating the os image versions, update this variable.
# Also please update corresponding image for node e2e at:
# https://github.com/kubernetes/kubernetes/blob/master/test/e2e_node/jenkins/image-config.yaml
#
# By default, the latest image from the image family will be used unless an
# explicit image will be set.
GCI_VERSION=${KUBE_GCI_VERSION:-}
IMAGE_FAMILY=${KUBE_IMAGE_FAMILY:-cos-109-lts}
export MASTER_IMAGE=${KUBE_GCE_MASTER_IMAGE:-}
export MASTER_IMAGE_FAMILY=${KUBE_GCE_MASTER_IMAGE_FAMILY:-${IMAGE_FAMILY}}
export MASTER_IMAGE_PROJECT=${KUBE_GCE_MASTER_PROJECT:-cos-cloud}
export NODE_IMAGE=${KUBE_GCE_NODE_IMAGE:-${GCI_VERSION}}
export NODE_IMAGE_FAMILY=${KUBE_GCE_NODE_IMAGE_FAMILY:-${IMAGE_FAMILY}}
export NODE_IMAGE_PROJECT=${KUBE_GCE_NODE_PROJECT:-cos-cloud}
export NODE_SERVICE_ACCOUNT=${KUBE_GCE_NODE_SERVICE_ACCOUNT:-default}

# KUBELET_TEST_ARGS are extra arguments passed to kubelet.
export KUBELET_TEST_ARGS=${KUBE_KUBELET_EXTRA_ARGS:-}

export CONTAINER_RUNTIME_ENDPOINT=${KUBE_CONTAINER_RUNTIME_ENDPOINT:-unix:///run/containerd/containerd.sock}
export CONTAINER_RUNTIME_NAME=${KUBE_CONTAINER_RUNTIME_NAME:-containerd}
export LOG_DUMP_SYSTEMD_SERVICES=${LOG_DUMP_SYSTEMD_SERVICES:-containerd}
export LOAD_IMAGE_COMMAND=${KUBE_LOAD_IMAGE_COMMAND:-ctr -n=k8s.io images import}

# Ability to inject custom versions (Ubuntu OS images ONLY)
# if KUBE_UBUNTU_INSTALL_CONTAINERD_VERSION or KUBE_UBUNTU_INSTALL_RUNC_VERSION
# is set to empty then we do not override the version(s) and just
# use whatever is in the default installation of containerd package
export UBUNTU_INSTALL_CONTAINERD_VERSION=${KUBE_UBUNTU_INSTALL_CONTAINERD_VERSION:-}
export UBUNTU_INSTALL_RUNC_VERSION=${KUBE_UBUNTU_INSTALL_RUNC_VERSION:-}

# Ability to inject custom versions (COS images ONLY)
# if KUBE_COS_INSTALL_CONTAINERD_VERSION or KUBE_COS_INSTALL_RUNC_VERSION
# is set to empty then we do not override the version(s) and just
# use whatever is in the default installation of containerd package
export COS_INSTALL_CONTAINERD_VERSION=${KUBE_COS_INSTALL_CONTAINERD_VERSION:-}
export COS_INSTALL_RUNC_VERSION=${KUBE_COS_INSTALL_RUNC_VERSION:-}

# MASTER_EXTRA_METADATA is the extra instance metadata on master instance separated by commas.
export MASTER_EXTRA_METADATA=${KUBE_MASTER_EXTRA_METADATA:-${KUBE_EXTRA_METADATA:-}}
# MASTER_EXTRA_METADATA is the extra instance metadata on node instance separated by commas.
export NODE_EXTRA_METADATA=${KUBE_NODE_EXTRA_METADATA:-${KUBE_EXTRA_METADATA:-}}

NETWORK=${KUBE_GCE_NETWORK:-default}
# Enable network deletion by default (for kube-down), unless we're using 'default' network.
if [[ "${NETWORK}" == "default" ]]; then
  KUBE_DELETE_NETWORK=${KUBE_DELETE_NETWORK:-false}
else
  KUBE_DELETE_NETWORK=${KUBE_DELETE_NETWORK:-true}
fi
if [[ "${CREATE_CUSTOM_NETWORK}" == true ]]; then
  SUBNETWORK="${SUBNETWORK:-${NETWORK}-custom-subnet}"
fi
INSTANCE_PREFIX="${KUBE_GCE_INSTANCE_PREFIX:-kubernetes}"
CLUSTER_NAME="${CLUSTER_NAME:-${INSTANCE_PREFIX}}"
MASTER_NAME="${INSTANCE_PREFIX}-master"
export AGGREGATOR_MASTER_NAME="${INSTANCE_PREFIX}-aggregator"
export INITIAL_ETCD_CLUSTER="${MASTER_NAME}"
export MASTER_TAG="${INSTANCE_PREFIX}-master"
export NODE_TAG="${INSTANCE_PREFIX}-minion"

CLUSTER_IP_RANGE="${CLUSTER_IP_RANGE:-$(get-cluster-ip-range)}"
MASTER_IP_RANGE="${MASTER_IP_RANGE:-10.246.0.0/24}"
# NODE_IP_RANGE is used when ENABLE_IP_ALIASES=true or CREATE_CUSTOM_NETWORK=true.
# It is the primary range in the subnet and is the range used for node instance IPs.
NODE_IP_RANGE="$(get-node-ip-range)"
export NODE_IP_RANGE

# NOTE: Avoid giving nodes empty scopes, because kubelet needs a service account
# in order to initialize properly.
NODE_SCOPES="${NODE_SCOPES:-monitoring,logging-write,storage-ro}"

# Extra docker options for nodes.
EXTRA_DOCKER_OPTS="${EXTRA_DOCKER_OPTS:-}"

VOLUME_PLUGIN_DIR="${VOLUME_PLUGIN_DIR:-/home/kubernetes/flexvolume}"

SERVICE_CLUSTER_IP_RANGE="${SERVICE_CLUSTER_IP_RANGE:-10.0.0.0/16}"  # formerly PORTAL_NET
export ALLOCATE_NODE_CIDRS=true

# When set to true, Docker Cache is enabled by default as part of the cluster bring up.
export ENABLE_DOCKER_REGISTRY_CACHE=true

# Optional: Deploy a L7 loadbalancer controller to fulfill Ingress requests:
#   glbc           - CE L7 Load Balancer Controller
export ENABLE_L7_LOADBALANCING="${KUBE_ENABLE_L7_LOADBALANCING:-glbc}"

# Optional: Enable Metrics Server. Metrics Server should be enable everywhere,
# since it's a critical component, but in the first release we need a way to disable
# this in case of stability issues.
# TODO(piosz) remove this option once Metrics Server became a stable thing.
export ENABLE_METRICS_SERVER="${KUBE_ENABLE_METRICS_SERVER:-true}"

# Optional: Metadata agent to setup as part of the cluster bring up:
#   none        - No metadata agent
#   stackdriver - Stackdriver metadata agent
# Metadata agent is a daemon set that provides metadata of kubernetes objects
# running on the same node for exporting metrics and logs.
export ENABLE_METADATA_AGENT="${KUBE_ENABLE_METADATA_AGENT:-none}"

# One special node out of NUM_NODES would be created of this type if specified.
# Useful for scheduling heapster in large clusters with nodes of small size.
HEAPSTER_MACHINE_TYPE="${HEAPSTER_MACHINE_TYPE:-}"

# Optional: Additional nodes would be created if their type and number is specified.
# NUM_NODES would be lowered respectively.
# Useful for running cluster-level addons that needs more resources than would fit
# on small nodes, like network plugins.
NUM_ADDITIONAL_NODES="${NUM_ADDITIONAL_NODES:-}"
ADDITIONAL_MACHINE_TYPE="${ADDITIONAL_MACHINE_TYPE:-}"

export MASTER_NODE_LABELS="${KUBE_MASTER_NODE_LABELS:-}"
# NON_MASTER_NODE_LABELS are labels will only be applied on non-master nodes.
NON_MASTER_NODE_LABELS="${KUBE_NON_MASTER_NODE_LABELS:-}"
WINDOWS_NON_MASTER_NODE_LABELS="${WINDOWS_NON_MASTER_NODE_LABELS:-}"

if [[ "${PREEMPTIBLE_MASTER}" == "true" ]]; then
    NODE_LABELS="${NODE_LABELS},cloud.google.com/gke-preemptible=true"
    WINDOWS_NODE_LABELS="${WINDOWS_NODE_LABELS},cloud.google.com/gke-preemptible=true"
elif [[ "${PREEMPTIBLE_NODE}" == "true" ]]; then
    NON_MASTER_NODE_LABELS="${NON_MASTER_NODE_LABELS},cloud.google.com/gke-preemptible=true"
    WINDOWS_NON_MASTER_NODE_LABELS="${WINDOWS_NON_MASTER_NODE_LABELS},cloud.google.com/gke-preemptible=true"
fi

# To avoid running Calico on a node that is not configured appropriately,
# label each Node so that the DaemonSet can run the Pods only on ready Nodes.
# Windows nodes do not support Calico.
if [[ ${NETWORK_POLICY_PROVIDER:-} == "calico" ]]; then
	NON_MASTER_NODE_LABELS="${NON_MASTER_NODE_LABELS:+${NON_MASTER_NODE_LABELS},}projectcalico.org/ds-ready=true"
fi

# Optional: Enable netd.
ENABLE_NETD="${KUBE_ENABLE_NETD:-false}"
export CUSTOM_NETD_YAML="${KUBE_CUSTOM_NETD_YAML:-}"
export CUSTOM_CALICO_NODE_DAEMONSET_YAML="${KUBE_CUSTOM_CALICO_NODE_DAEMONSET_YAML:-}"
export CUSTOM_TYPHA_DEPLOYMENT_YAML="${KUBE_CUSTOM_TYPHA_DEPLOYMENT_YAML:-}"

# To avoid running netd on a node that is not configured appropriately,
# label each Node so that the DaemonSet can run the Pods only on ready Nodes.
# Windows nodes do not support netd.
if [[ ${ENABLE_NETD:-} == "true" ]]; then
	NON_MASTER_NODE_LABELS="${NON_MASTER_NODE_LABELS:+${NON_MASTER_NODE_LABELS},}cloud.google.com/gke-netd-ready=true"
fi

export ENABLE_NODELOCAL_DNS="${KUBE_ENABLE_NODELOCAL_DNS:-false}"
export LOCAL_DNS_IP="${KUBE_LOCAL_DNS_IP:-169.254.20.10}"

# Enable metadata concealment by firewalling pod traffic to the metadata server
# and run a proxy daemonset on nodes.
#
# TODO(#8867) Enable by default.
ENABLE_METADATA_CONCEALMENT="${ENABLE_METADATA_CONCEALMENT:-false}" # true, false
METADATA_CONCEALMENT_NO_FIREWALL="${METADATA_CONCEALMENT_NO_FIREWALL:-false}" # true, false
if [[ ${ENABLE_METADATA_CONCEALMENT:-} == "true" ]]; then
  # Put the necessary label on the node so the daemonset gets scheduled.
  NODE_LABELS="${NODE_LABELS},cloud.google.com/metadata-proxy-ready=true"
  # Add to the provider custom variables.
  PROVIDER_VARS="${PROVIDER_VARS:-} ENABLE_METADATA_CONCEALMENT METADATA_CONCEALMENT_NO_FIREWALL"
fi

# Optional: Enable node logging.
export ENABLE_NODE_LOGGING="${KUBE_ENABLE_NODE_LOGGING:-true}"
export LOGGING_DESTINATION="${KUBE_LOGGING_DESTINATION:-gcp}" # options: gcp

# Optional: When set to true, Elasticsearch and Kibana will be setup as part of the cluster bring up.
export ENABLE_CLUSTER_LOGGING="${KUBE_ENABLE_CLUSTER_LOGGING:-true}"
export ELASTICSEARCH_LOGGING_REPLICAS=1

# Optional: Don't require https for registries in our local RFC1918 network
if [[ ${KUBE_ENABLE_INSECURE_REGISTRY:-false} == "true" ]]; then
  EXTRA_DOCKER_OPTS="${EXTRA_DOCKER_OPTS} --insecure-registry 10.0.0.0/8"
fi

# Optional: customize runtime config
RUNTIME_CONFIG="${KUBE_RUNTIME_CONFIG:-}"

if [[ "${KUBE_FEATURE_GATES:-}" == "AllAlpha=true" ]]; then
  export RUNTIME_CONFIG="${KUBE_RUNTIME_CONFIG:-api/all=true}"
fi

# By default disable gkenetworkparamset controller in CCM
RUN_CCM_CONTROLLERS="${RUN_CCM_CONTROLLERS:-*,-gkenetworkparamset}"

# List of the set of feature gates recognized by the GCP CCM
export CCM_FEATURE_GATES="APIPriorityAndFairness,APIResponseCompression,APIServerIdentity,APIServerTracing,AllAlpha,AllBeta,CustomResourceValidationExpressions,KMSv2,OpenAPIEnums,OpenAPIV3,ServerSideFieldValidation,StorageVersionAPI,StorageVersionHash"

# Optional: set feature gates
# shellcheck disable=SC2034 # Variables sourced in other scripts.
FEATURE_GATES="${KUBE_FEATURE_GATES:-}"

if [[ -n "${NODE_ACCELERATORS}" ]]; then
    if [[ "${NODE_ACCELERATORS}" =~ .*type=([a-zA-Z0-9-]+).* ]]; then
        NON_MASTER_NODE_LABELS="${NON_MASTER_NODE_LABELS},cloud.google.com/gke-accelerator=${BASH_REMATCH[1]}"
    fi
fi

# Optional: Install cluster DNS.
# Set CLUSTER_DNS_CORE_DNS to 'false' to install kube-dns instead of CoreDNS.
CLUSTER_DNS_CORE_DNS="${CLUSTER_DNS_CORE_DNS:-true}"
export ENABLE_CLUSTER_DNS="${KUBE_ENABLE_CLUSTER_DNS:-true}"
export DNS_SERVER_IP="${KUBE_DNS_SERVER_IP:-10.0.0.10}"
export DNS_DOMAIN="${KUBE_DNS_DOMAIN:-cluster.local}"
export DNS_MEMORY_LIMIT="${KUBE_DNS_MEMORY_LIMIT:-170Mi}"

# Optional: Enable DNS horizontal autoscaler
export ENABLE_DNS_HORIZONTAL_AUTOSCALER="${KUBE_ENABLE_DNS_HORIZONTAL_AUTOSCALER:-true}"

# Optional: Install node problem detector.
#   none           - Not run node problem detector.
#   daemonset      - Run node problem detector as daemonset.
#   standalone     - Run node problem detector as standalone system daemon.
export ENABLE_NODE_PROBLEM_DETECTOR="${KUBE_ENABLE_NODE_PROBLEM_DETECTOR:-daemonset}"
NODE_PROBLEM_DETECTOR_VERSION="${NODE_PROBLEM_DETECTOR_VERSION:-}"
NODE_PROBLEM_DETECTOR_TAR_HASH="${NODE_PROBLEM_DETECTOR_TAR_HASH:-}"
NODE_PROBLEM_DETECTOR_RELEASE_PATH="${NODE_PROBLEM_DETECTOR_RELEASE_PATH:-}"
NODE_PROBLEM_DETECTOR_CUSTOM_FLAGS="${NODE_PROBLEM_DETECTOR_CUSTOM_FLAGS:-}"

CNI_HASH="${CNI_HASH:-}"
CNI_TAR_PREFIX="${CNI_TAR_PREFIX:-cni-plugins-linux-amd64-}"
CNI_STORAGE_URL_BASE="${CNI_STORAGE_URL_BASE:-https://storage.googleapis.com/k8s-artifacts-cni/release}"

# Optional: Create autoscaler for cluster's nodes.
ENABLE_CLUSTER_AUTOSCALER="${KUBE_ENABLE_CLUSTER_AUTOSCALER:-false}"
if [[ "${ENABLE_CLUSTER_AUTOSCALER}" == "true" ]]; then
  export AUTOSCALER_MIN_NODES="${KUBE_AUTOSCALER_MIN_NODES:-}"
  export AUTOSCALER_MAX_NODES="${KUBE_AUTOSCALER_MAX_NODES:-}"
  export AUTOSCALER_ENABLE_SCALE_DOWN="${KUBE_AUTOSCALER_ENABLE_SCALE_DOWN:-true}"
  export AUTOSCALER_EXPANDER_CONFIG="${KUBE_AUTOSCALER_EXPANDER_CONFIG:---expander=price}"
fi

# Optional: Enable allocation of pod IPs using IP aliases.
#
# BETA FEATURE.
#
# IP_ALIAS_SIZE is the size of the podCIDR allocated to a node.
# IP_ALIAS_SUBNETWORK is the subnetwork to allocate from. If empty, a
#   new subnetwork will be created for the cluster.
ENABLE_IP_ALIASES=${KUBE_GCE_ENABLE_IP_ALIASES:-true}
NODE_IPAM_MODE=${KUBE_GCE_NODE_IPAM_MODE:-RangeAllocator}
if [ "${ENABLE_IP_ALIASES}" = true ]; then
  # Number of Pods that can run on this node.
  MAX_PODS_PER_NODE=${MAX_PODS_PER_NODE:-110}
  # Size of ranges allocated to each node.
  IP_ALIAS_SIZE="/$(get-alias-range-size "${MAX_PODS_PER_NODE}")"
  export IP_ALIAS_SIZE
  IP_ALIAS_SUBNETWORK=${KUBE_GCE_IP_ALIAS_SUBNETWORK:-${INSTANCE_PREFIX}-subnet-default}
  # If we're using custom network, use the subnet we already create for it as the one for ip-alias.
  # Note that this means SUBNETWORK would override KUBE_GCE_IP_ALIAS_SUBNETWORK in case of custom network.
  if [[ "${CREATE_CUSTOM_NETWORK}" == true ]]; then
    export IP_ALIAS_SUBNETWORK="${SUBNETWORK}"
  fi
  export NODE_IPAM_MODE=${KUBE_GCE_NODE_IPAM_MODE:-CloudAllocator}
  SECONDARY_RANGE_NAME=${SECONDARY_RANGE_NAME:-}
  # Add to the provider custom variables.
  PROVIDER_VARS="${PROVIDER_VARS:-} ENABLE_IP_ALIASES"
  PROVIDER_VARS="${PROVIDER_VARS:-} NODE_IPAM_MODE"
  PROVIDER_VARS="${PROVIDER_VARS:-} SECONDARY_RANGE_NAME"
else
  if [[ -n "${MAX_PODS_PER_NODE:-}" ]]; then
    # Should not have MAX_PODS_PER_NODE set for route-based clusters.
    echo -e "${color_red:-}Cannot set MAX_PODS_PER_NODE for route-based projects for ${PROJECT}." >&2
    exit 1
  fi
  if [[ "$(get-num-nodes)" -gt 100 ]]; then
    echo -e "${color_red:-}Cannot create cluster with more than 100 nodes for route-based projects for ${PROJECT}." >&2
    exit 1
  fi
fi

# Enable GCE Alpha features.
if [[ -n "${GCE_ALPHA_FEATURES:-}" ]]; then
  PROVIDER_VARS="${PROVIDER_VARS:-} GCE_ALPHA_FEATURES"
fi

# Disable Docker live-restore.
if [[ -n "${DISABLE_DOCKER_LIVE_RESTORE:-}" ]]; then
  PROVIDER_VARS="${PROVIDER_VARS:-} DISABLE_DOCKER_LIVE_RESTORE"
fi

# Override default GLBC image
if [[ -n "${GCE_GLBC_IMAGE:-}" ]]; then
  PROVIDER_VARS="${PROVIDER_VARS:-} GCE_GLBC_IMAGE"
fi
CUSTOM_INGRESS_YAML="${CUSTOM_INGRESS_YAML:-}"

# Admission Controllers to invoke prior to persisting objects in cluster
ADMISSION_CONTROL=NamespaceLifecycle,LimitRanger,ServiceAccount,DefaultStorageClass,PersistentVolumeClaimResize,DefaultTolerationSeconds,NodeRestriction,Priority,StorageObjectInUseProtection,RuntimeClass

# MutatingAdmissionWebhook should be the last controller that modifies the
# request object, otherwise users will be confused if the mutating webhooks'
# modification is overwritten.
ADMISSION_CONTROL="${ADMISSION_CONTROL},MutatingAdmissionWebhook,ValidatingAdmissionWebhook"

# ResourceQuota must come last, or a creation is recorded, but the pod was forbidden.
ADMISSION_CONTROL="${ADMISSION_CONTROL},ResourceQuota"

# Optional: if set to true kube-up will automatically check for existing resources and clean them up.
KUBE_UP_AUTOMATIC_CLEANUP=${KUBE_UP_AUTOMATIC_CLEANUP:-false}

# Storage backend. 'etcd2' supported, 'etcd3' experimental.
STORAGE_BACKEND=${STORAGE_BACKEND:-}

# Networking plugin specific settings.
NETWORK_PROVIDER="${NETWORK_PROVIDER:-kubenet}" # none, kubenet

# Network Policy plugin specific settings.
NETWORK_POLICY_PROVIDER="${NETWORK_POLICY_PROVIDER:-none}" # calico

export NON_MASQUERADE_CIDR="0.0.0.0/0"

# How should the kubelet configure hairpin mode?
HAIRPIN_MODE="${HAIRPIN_MODE:-hairpin-veth}" # promiscuous-bridge, hairpin-veth, none
# Optional: if set to true, kube-up will configure the cluster to run e2e tests.
export E2E_STORAGE_TEST_ENVIRONMENT="${KUBE_E2E_STORAGE_TEST_ENVIRONMENT:-false}"

# Evict pods whenever compute resource availability on the nodes gets below a threshold.
EVICTION_HARD="${EVICTION_HARD:-memory.available<250Mi,nodefs.available<10%,nodefs.inodesFree<5%}"

# Optional: custom scheduling algorithm
SCHEDULING_ALGORITHM_PROVIDER="${SCHEDULING_ALGORITHM_PROVIDER:-}"

# Optional: install a default StorageClass
ENABLE_DEFAULT_STORAGE_CLASS="${ENABLE_DEFAULT_STORAGE_CLASS:-false}"

# Optional: install volume snapshot CRDs
ENABLE_VOLUME_SNAPSHOTS="${ENABLE_VOLUME_SNAPSHOTS:-true}"

# Optional: Enable legacy ABAC policy that makes all service accounts superusers.
ENABLE_LEGACY_ABAC="${ENABLE_LEGACY_ABAC:-false}" # true, false

# Indicates if the values (i.e. KUBE_USER and KUBE_PASSWORD for basic
# authentication) in metadata should be treated as canonical, and therefore disk
# copies ought to be recreated/clobbered.
METADATA_CLOBBERS_CONFIG="${METADATA_CLOBBERS_CONFIG:-false}"

ENABLE_BIG_CLUSTER_SUBNETS="${ENABLE_BIG_CLUSTER_SUBNETS:-false}"

# Optional: Enable log rotation for k8s services
ENABLE_LOGROTATE_FILES="${ENABLE_LOGROTATE_FILES:-true}"
PROVIDER_VARS="${PROVIDER_VARS:-} ENABLE_LOGROTATE_FILES"
if [[ -n "${LOGROTATE_FILES_MAX_COUNT:-}" ]]; then
  PROVIDER_VARS="${PROVIDER_VARS:-} LOGROTATE_FILES_MAX_COUNT"
fi
if [[ -n "${LOGROTATE_MAX_SIZE:-}" ]]; then
  PROVIDER_VARS="${PROVIDER_VARS:-} LOGROTATE_MAX_SIZE"
fi

# Optional: Enable log rotation for pod logs
ENABLE_POD_LOG="${ENABLE_POD_LOG:-false}"
PROVIDER_VARS="${PROVIDER_VARS:-} ENABLE_POD_LOG"

if [[ -n "${POD_LOG_MAX_FILE:-}" ]]; then
  PROVIDER_VARS="${PROVIDER_VARS:-} POD_LOG_MAX_FILE"
fi

if [[ -n "${POD_LOG_MAX_SIZE:-}" ]]; then
  PROVIDER_VARS="${PROVIDER_VARS:-} POD_LOG_MAX_SIZE"
fi

# Fluentd requirements
# YAML exists to trigger a configuration refresh when changes are made.
export FLUENTD_GCP_YAML_VERSION="v3.2.0"
FLUENTD_GCP_VERSION="${FLUENTD_GCP_VERSION:-1.6.17}"
FLUENTD_GCP_MEMORY_LIMIT="${FLUENTD_GCP_MEMORY_LIMIT:-}"
FLUENTD_GCP_CPU_REQUEST="${FLUENTD_GCP_CPU_REQUEST:-}"
FLUENTD_GCP_MEMORY_REQUEST="${FLUENTD_GCP_MEMORY_REQUEST:-}"

# Heapster requirements
HEAPSTER_GCP_BASE_MEMORY="${HEAPSTER_GCP_BASE_MEMORY:-140Mi}"
HEAPSTER_GCP_MEMORY_PER_NODE="${HEAPSTER_GCP_MEMORY_PER_NODE:-4}"
HEAPSTER_GCP_BASE_CPU="${HEAPSTER_GCP_BASE_CPU:-80m}"
HEAPSTER_GCP_CPU_PER_NODE="${HEAPSTER_GCP_CPU_PER_NODE:-0.5}"

# Default Stackdriver resources version exported by Fluentd-gcp addon
LOGGING_STACKDRIVER_RESOURCE_TYPES="${LOGGING_STACKDRIVER_RESOURCE_TYPES:-old}"

# Adding to PROVIDER_VARS, since this is GCP-specific.
PROVIDER_VARS="${PROVIDER_VARS:-} FLUENTD_GCP_YAML_VERSION FLUENTD_GCP_VERSION FLUENTD_GCP_MEMORY_LIMIT FLUENTD_GCP_CPU_REQUEST FLUENTD_GCP_MEMORY_REQUEST HEAPSTER_GCP_BASE_MEMORY HEAPSTER_GCP_MEMORY_PER_NODE HEAPSTER_GCP_BASE_CPU HEAPSTER_GCP_CPU_PER_NODE LOGGING_STACKDRIVER_RESOURCE_TYPES"

# Fluentd configuration for node-journal
ENABLE_NODE_JOURNAL="${ENABLE_NODE_JOURNAL:-false}"

# prometheus-to-sd configuration
PROMETHEUS_TO_SD_ENDPOINT="${PROMETHEUS_TO_SD_ENDPOINT:-https://monitoring.googleapis.com/}"
PROMETHEUS_TO_SD_PREFIX="${PROMETHEUS_TO_SD_PREFIX:-custom.googleapis.com}"
ENABLE_PROMETHEUS_TO_SD="${ENABLE_PROMETHEUS_TO_SD:-false}"

# TODO(#51292): Make kube-proxy Daemonset default and remove the configuration here.
# Optional: [Experiment Only] Run kube-proxy as a DaemonSet if set to true, run as static pods otherwise.
KUBE_PROXY_DAEMONSET="${KUBE_PROXY_DAEMONSET:-false}" # true, false

# Control whether the startup scripts manage the lifecycle of kube-proxy
# When true, the startup scripts do not enable kube-proxy either as a daemonset addon or as a static pod
# regardless of the value of KUBE_PROXY_DAEMONSET.
# When false, the value of KUBE_PROXY_DAEMONSET controls whether kube-proxy comes up as a static pod or
# as an addon daemonset.
KUBE_PROXY_DISABLE="${KUBE_PROXY_DISABLE:-false}" # true, false

# Will be passed into the kube-proxy via `--detect-local-mode`
DETECT_LOCAL_MODE="${DETECT_LOCAL_MODE:-}"

# Optional: duration of cluster signed certificates.
CLUSTER_SIGNING_DURATION="${CLUSTER_SIGNING_DURATION:-}"

# Optional: enable certificate rotation of the kubelet certificates.
ROTATE_CERTIFICATES="${ROTATE_CERTIFICATES:-}"

# The number of services that are allowed to sync concurrently. Will be passed
# into kube-controller-manager via `--concurrent-service-syncs`
CONCURRENT_SERVICE_SYNCS="${CONCURRENT_SERVICE_SYNCS:-5}"

export SERVICEACCOUNT_ISSUER="https://kubernetes.io/${CLUSTER_NAME}"

# Taint Windows nodes by default to prevent Linux workloads from being
# scheduled onto them.
WINDOWS_NODE_TAINTS="${WINDOWS_NODE_TAINTS:-node.kubernetes.io/os=win1809:NoSchedule}"

# Whether to set up a private GCE cluster, i.e. a cluster where nodes have only private IPs.
export GCE_PRIVATE_CLUSTER="${KUBE_GCE_PRIVATE_CLUSTER:-false}"
export GCE_PRIVATE_CLUSTER_PORTS_PER_VM="${KUBE_GCE_PRIVATE_CLUSTER_PORTS_PER_VM:-}"

# When KUBE_ENABLE_KONNECTIVITY_SERVICE is enabled, the three variables below will
# default to true to enable the konnectivity network proxy and start the required pods.
# Their values can be overridden for more granular control of the proxy.

# Optional: Whether to do the setup for the konnectivity service
# Includes setting up kubeconfig, tokens, egress files, and firewall rules
export PREPARE_KONNECTIVITY_SERVICE="${KUBE_ENABLE_KONNECTIVITY_SERVICE:-true}"
# Optional: Whether to use konnectivity network proxy for all egress from apiserver.
export EGRESS_VIA_KONNECTIVITY="${KUBE_ENABLE_KONNECTIVITY_SERVICE:-true}"
# Optional: Whether to start the konnectivity server and agent pods.
export RUN_KONNECTIVITY_PODS="${KUBE_ENABLE_KONNECTIVITY_SERVICE:-true}"
# Proxy Protocol Mode determines the protocol to use to communicate between apiserver and network proxy.
# Valid options are grpc and http-connect. Default is grpc.
export KONNECTIVITY_SERVICE_PROXY_PROTOCOL_MODE="${KUBE_KONNECTIVITY_SERVICE_PROXY_PROTOCOL_MODE:-grpc}"

# Optional: Enable Windows CSI-Proxy
export ENABLE_CSI_PROXY="${ENABLE_CSI_PROXY:-true}"

# KUBE_APISERVER_HEALTHCHECK_ON_HOST_IP decides whether
# kube-apiserver is healthchecked on host IP instead of 127.0.0.1.
export KUBE_APISERVER_HEALTHCHECK_ON_HOST_IP="${KUBE_APISERVER_HEALTHCHECK_ON_HOST_IP:-false}"

# ETCD_PROGRESS_NOTIFY_INTERVAL defines the interval for etcd watch progress notify events.
export ETCD_PROGRESS_NOTIFY_INTERVAL="${ETCD_PROGRESS_NOTIFY_INTERVAL:-5s}"

# Optional: Install Pigz on Windows.
# Pigz is a multi-core optimized version of unzip.exe.
# It improves container image pull performance since most time is spent
# unzipping the image layers to disk.
export WINDOWS_ENABLE_PIGZ="${WINDOWS_ENABLE_PIGZ:-true}"

# Enable Windows DSR (Direct Server Return)
export WINDOWS_ENABLE_DSR="${WINDOWS_ENABLE_DSR:-false}"

# Install Node Problem Detector (NPD) on Windows nodes.
# NPD analyzes the host for problems that can disrupt workloads.
export WINDOWS_ENABLE_NODE_PROBLEM_DETECTOR="${WINDOWS_ENABLE_NODE_PROBLEM_DETECTOR:-none}"
export WINDOWS_NODE_PROBLEM_DETECTOR_CUSTOM_FLAGS="${WINDOWS_NODE_PROBLEM_DETECTOR_CUSTOM_FLAGS:-}"

# Enable Windows Hyper-V
# sig-storage uses it to create Virtual Hard Disks in tests
export WINDOWS_ENABLE_HYPERV="${WINDOWS_ENABLE_HYPERV:-false}"

# TLS_CIPHER_SUITES defines cipher suites allowed to be used by kube-apiserver.
# If this variable is unset or empty, kube-apiserver will allow its default set of cipher suites.
export TLS_CIPHER_SUITES=""

# CLOUD_PROVIDER_FLAG defines the cloud-provider value presented to KCM, apiserver,
# and kubelet
export CLOUD_PROVIDER_FLAG="${CLOUD_PROVIDER_FLAG:-external}"

# Don't run the node-ipam-controller on the KCM if cloud-provider external
if [[ "${CLOUD_PROVIDER_FLAG}" ==  "external" ]]; then
  RUN_CONTROLLERS="${RUN_CONTROLLERS:-*,-node-ipam-controller}"
fi

# When ENABLE_AUTH_PROVIDER_GCP is set, following flags for out-of-tree credential provider for GCP
# are presented to kubelet:
# --image-credential-provider-config=${path-to-config}
# --image-credential-provider-bin-dir=${path-to-auth-provider-binary}
# Also, it is required that DisableKubeletCloudCredentialProviders
# feature gates are set to true for kubelet to use external credential provider.
export ENABLE_AUTH_PROVIDER_GCP="${ENABLE_AUTH_PROVIDER_GCP:-true}"
