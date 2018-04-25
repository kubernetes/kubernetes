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

# TODO(jbeda): Provide a way to override project
# gcloud multiplexing for shared GCE/GKE tests.
KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/cluster/gce/config-common.sh"

# Specifying KUBE_GCE_API_ENDPOINT will override the default GCE Compute API endpoint (https://www.googleapis.com/compute/v1/).
# This endpoint has to be pointing to v1 api. For example, https://www.googleapis.com/compute/staging_v1/
GCE_API_ENDPOINT=${KUBE_GCE_API_ENDPOINT:-}
GCLOUD=gcloud
ZONE=${KUBE_GCE_ZONE:-us-central1-b}
REGION=${ZONE%-*}
RELEASE_REGION_FALLBACK=${RELEASE_REGION_FALLBACK:-false}
REGIONAL_KUBE_ADDONS=${REGIONAL_KUBE_ADDONS:-true}
NODE_SIZE=${NODE_SIZE:-n1-standard-2}
NUM_NODES=${NUM_NODES:-3}
MASTER_SIZE=${MASTER_SIZE:-n1-standard-$(get-master-size)}
MASTER_MIN_CPU_ARCHITECTURE=${MASTER_MIN_CPU_ARCHITECTURE:-} # To allow choosing better architectures.
MASTER_DISK_TYPE=pd-ssd
MASTER_DISK_SIZE=${MASTER_DISK_SIZE:-$(get-master-disk-size)}
MASTER_ROOT_DISK_SIZE=${MASTER_ROOT_DISK_SIZE:-$(get-master-root-disk-size)}
NODE_DISK_TYPE=${NODE_DISK_TYPE:-pd-standard}
NODE_DISK_SIZE=${NODE_DISK_SIZE:-100GB}
NODE_LOCAL_SSDS=${NODE_LOCAL_SSDS:-0}
# An extension to local SSDs allowing users to specify block/fs and SCSI/NVMe devices
# Format of this variable will be "#,scsi/nvme,block/fs" you can specify multiple
# configurations by separating them by a semi-colon ex. "2,scsi,fs;1,nvme,block"
# is a request for 2 SCSI formatted and mounted SSDs and 1 NVMe block device SSD.
NODE_LOCAL_SSDS_EXT=${NODE_LOCAL_SSDS_EXT:-}
NODE_ACCELERATORS=${NODE_ACCELERATORS:-""}
REGISTER_MASTER_KUBELET=${REGISTER_MASTER:-true}
KUBE_APISERVER_REQUEST_TIMEOUT=300
PREEMPTIBLE_NODE=${PREEMPTIBLE_NODE:-false}
PREEMPTIBLE_MASTER=${PREEMPTIBLE_MASTER:-false}
KUBE_DELETE_NODES=${KUBE_DELETE_NODES:-true}
KUBE_DELETE_NETWORK=${KUBE_DELETE_NETWORK:-true}
CREATE_CUSTOM_NETWORK=${CREATE_CUSTOM_NETWORK:-false}

MASTER_OS_DISTRIBUTION=${KUBE_MASTER_OS_DISTRIBUTION:-${KUBE_OS_DISTRIBUTION:-gci}}
NODE_OS_DISTRIBUTION=${KUBE_NODE_OS_DISTRIBUTION:-${KUBE_OS_DISTRIBUTION:-gci}}
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
CVM_VERSION=${CVM_VERSION:-container-vm-v20170627}
GCI_VERSION=${KUBE_GCI_VERSION:-cos-stable-63-10032-71-0}
MASTER_IMAGE=${KUBE_GCE_MASTER_IMAGE:-}
MASTER_IMAGE_PROJECT=${KUBE_GCE_MASTER_PROJECT:-cos-cloud}
NODE_IMAGE=${KUBE_GCE_NODE_IMAGE:-${GCI_VERSION}}
NODE_IMAGE_PROJECT=${KUBE_GCE_NODE_PROJECT:-cos-cloud}
NODE_SERVICE_ACCOUNT=${KUBE_GCE_NODE_SERVICE_ACCOUNT:-default}
CONTAINER_RUNTIME=${KUBE_CONTAINER_RUNTIME:-docker}
CONTAINER_RUNTIME_ENDPOINT=${KUBE_CONTAINER_RUNTIME_ENDPOINT:-}
CONTAINER_RUNTIME_NAME=${KUBE_CONTAINER_RUNTIME_NAME:-}
LOAD_IMAGE_COMMAND=${KUBE_LOAD_IMAGE_COMMAND:-}
GCI_DOCKER_VERSION=${KUBE_GCI_DOCKER_VERSION:-}
RKT_VERSION=${KUBE_RKT_VERSION:-1.23.0}
RKT_STAGE1_IMAGE=${KUBE_RKT_STAGE1_IMAGE:-coreos.com/rkt/stage1-coreos}
# MASTER_EXTRA_METADATA is the extra instance metadata on master instance separated by commas.
MASTER_EXTRA_METADATA=${KUBE_MASTER_EXTRA_METADATA:-${KUBE_EXTRA_METADATA:-}}
# MASTER_EXTRA_METADATA is the extra instance metadata on node instance separated by commas.
NODE_EXTRA_METADATA=${KUBE_NODE_EXTRA_METADATA:-${KUBE_EXTRA_METADATA:-}}

NETWORK=${KUBE_GCE_NETWORK:-e2e-test-${USER}}
if [[ "${CREATE_CUSTOM_NETWORK}" == true ]]; then
  SUBNETWORK="${SUBNETWORK:-${NETWORK}-custom-subnet}"
fi
INSTANCE_PREFIX="${KUBE_GCE_INSTANCE_PREFIX:-e2e-test-${USER}}"
CLUSTER_NAME="${CLUSTER_NAME:-${INSTANCE_PREFIX}}"
MASTER_NAME="${INSTANCE_PREFIX}-master"
AGGREGATOR_MASTER_NAME="${INSTANCE_PREFIX}-aggregator"
INITIAL_ETCD_CLUSTER="${MASTER_NAME}"
ETCD_QUORUM_READ="${ENABLE_ETCD_QUORUM_READ:-false}"
MASTER_TAG="${INSTANCE_PREFIX}-master"
NODE_TAG="${INSTANCE_PREFIX}-minion"

CLUSTER_IP_RANGE="${CLUSTER_IP_RANGE:-$(get-cluster-ip-range)}"
MASTER_IP_RANGE="${MASTER_IP_RANGE:-10.246.0.0/24}"
# NODE_IP_RANGE is used when ENABLE_IP_ALIASES=true or CREATE_CUSTOM_NETWORK=true.
# It is the primary range in the subnet and is the range used for node instance IPs.
NODE_IP_RANGE="$(get-node-ip-range)"

RUNTIME_CONFIG="${KUBE_RUNTIME_CONFIG:-}"

# Optional: set feature gates
FEATURE_GATES="${KUBE_FEATURE_GATES:-ExperimentalCriticalPodAnnotation=true}"

TERMINATED_POD_GC_THRESHOLD=${TERMINATED_POD_GC_THRESHOLD:-100}

# Extra docker options for nodes.
EXTRA_DOCKER_OPTS="${EXTRA_DOCKER_OPTS:-}"

# Enable the docker debug mode.
EXTRA_DOCKER_OPTS="${EXTRA_DOCKER_OPTS} --debug"

SERVICE_CLUSTER_IP_RANGE="10.0.0.0/16"  # formerly PORTAL_NET

# When set to true, Docker Cache is enabled by default as part of the cluster bring up.
ENABLE_DOCKER_REGISTRY_CACHE=true

# Optional: Deploy a L7 loadbalancer controller to fulfill Ingress requests:
#   glbc           - CE L7 Load Balancer Controller
ENABLE_L7_LOADBALANCING="${KUBE_ENABLE_L7_LOADBALANCING:-glbc}"

# Optional: Cluster monitoring to setup as part of the cluster bring up:
#   none           - No cluster monitoring setup
#   influxdb       - Heapster, InfluxDB, and Grafana
#   google         - Heapster, Google Cloud Monitoring, and Google Cloud Logging
#   stackdriver    - Heapster, Google Cloud Monitoring (schema container), and Google Cloud Logging
#   googleinfluxdb - Enable influxdb and google (except GCM)
#   standalone     - Heapster only. Metrics available via Heapster REST API.
ENABLE_CLUSTER_MONITORING="${KUBE_ENABLE_CLUSTER_MONITORING:-influxdb}"

# Optional: Enable Metrics Server. Metrics Server should be enable everywhere,
# since it's a critical component, but in the first release we need a way to disable
# this in case of stability issues.
# TODO(piosz) remove this option once Metrics Server became a stable thing.
ENABLE_METRICS_SERVER="${KUBE_ENABLE_METRICS_SERVER:-true}"

# Optional: Metadata agent to setup as part of the cluster bring up:
#   none        - No metadata agent
#   stackdriver - Stackdriver metadata agent
# Metadata agent is a daemon set that provides metadata of kubernetes objects
# running on the same node for exporting metrics and logs.
ENABLE_METADATA_AGENT="${KUBE_ENABLE_METADATA_AGENT:-none}"

# One special node out of NUM_NODES would be created of this type if specified.
# Useful for scheduling heapster in large clusters with nodes of small size.
HEAPSTER_MACHINE_TYPE="${HEAPSTER_MACHINE_TYPE:-}"

# Set etcd image (e.g. k8s.gcr.io/etcd) and version (e.g. 3.1.12) if you need
# non-default version.
ETCD_IMAGE="${TEST_ETCD_IMAGE:-}"
ETCD_DOCKER_REPOSITORY="${TEST_ETCD_DOCKER_REPOSITORY:-}"
ETCD_VERSION="${TEST_ETCD_VERSION:-}"

# Default Log level for all components in test clusters and variables to override it in specific components.
TEST_CLUSTER_LOG_LEVEL="${TEST_CLUSTER_LOG_LEVEL:---v=4}"
KUBELET_TEST_LOG_LEVEL="${KUBELET_TEST_LOG_LEVEL:-$TEST_CLUSTER_LOG_LEVEL}"
DOCKER_TEST_LOG_LEVEL="${DOCKER_TEST_LOG_LEVEL:---log-level=info}"
API_SERVER_TEST_LOG_LEVEL="${API_SERVER_TEST_LOG_LEVEL:-$TEST_CLUSTER_LOG_LEVEL}"
CONTROLLER_MANAGER_TEST_LOG_LEVEL="${CONTROLLER_MANAGER_TEST_LOG_LEVEL:-$TEST_CLUSTER_LOG_LEVEL}"
SCHEDULER_TEST_LOG_LEVEL="${SCHEDULER_TEST_LOG_LEVEL:-$TEST_CLUSTER_LOG_LEVEL}"
KUBEPROXY_TEST_LOG_LEVEL="${KUBEPROXY_TEST_LOG_LEVEL:-$TEST_CLUSTER_LOG_LEVEL}"

VOLUME_PLUGIN_DIR="${VOLUME_PLUGIN_DIR:-/home/kubernetes/flexvolume}"

TEST_CLUSTER_DELETE_COLLECTION_WORKERS="${TEST_CLUSTER_DELETE_COLLECTION_WORKERS:---delete-collection-workers=1}"
TEST_CLUSTER_MAX_REQUESTS_INFLIGHT="${TEST_CLUSTER_MAX_REQUESTS_INFLIGHT:-}"
TEST_CLUSTER_RESYNC_PERIOD="${TEST_CLUSTER_RESYNC_PERIOD:---min-resync-period=3m}"

# ContentType used by all components to communicate with apiserver.
TEST_CLUSTER_API_CONTENT_TYPE="${TEST_CLUSTER_API_CONTENT_TYPE:-}"

KUBELET_TEST_ARGS="${KUBELET_TEST_ARGS:-} --max-pods=110 --serialize-image-pulls=false ${TEST_CLUSTER_API_CONTENT_TYPE}"
if [[ "${NODE_OS_DISTRIBUTION}" == "gci" ]] || [[ "${NODE_OS_DISTRIBUTION}" == "ubuntu" ]] || [[ "${NODE_OS_DISTRIBUTION}" == "custom" ]]; then
  NODE_KUBELET_TEST_ARGS=" --experimental-kernel-memcg-notification=true"
fi
if [[ "${MASTER_OS_DISTRIBUTION}" == "gci" ]] || [[ "${MASTER_OS_DISTRIBUTION}" == "ubuntu" ]]; then
  MASTER_KUBELET_TEST_ARGS=" --experimental-kernel-memcg-notification=true"
fi
APISERVER_TEST_ARGS="${APISERVER_TEST_ARGS:-} --runtime-config=extensions/v1beta1 ${TEST_CLUSTER_DELETE_COLLECTION_WORKERS} ${TEST_CLUSTER_MAX_REQUESTS_INFLIGHT}"
CONTROLLER_MANAGER_TEST_ARGS="${CONTROLLER_MANAGER_TEST_ARGS:-} ${TEST_CLUSTER_RESYNC_PERIOD} ${TEST_CLUSTER_API_CONTENT_TYPE}"
SCHEDULER_TEST_ARGS="${SCHEDULER_TEST_ARGS:-} ${TEST_CLUSTER_API_CONTENT_TYPE}"
KUBEPROXY_TEST_ARGS="${KUBEPROXY_TEST_ARGS:-} ${TEST_CLUSTER_API_CONTENT_TYPE}"

# Historically fluentd was a manifest pod and then was migrated to DaemonSet.
# To avoid situation during cluster upgrade when there are two instances
# of fluentd running on a node, kubelet need to mark node on which
# fluentd is not running as a manifest pod with appropriate label.
# TODO(piosz): remove this in 1.8
NODE_LABELS="${KUBE_NODE_LABELS:-beta.kubernetes.io/fluentd-ds-ready=true}"

# NON_MASTER_NODE_LABELS are labels will only be applied on non-master nodes.
NON_MASTER_NODE_LABELS="${KUBE_NON_MASTER_NODE_LABELS:-}"

# To avoid running Calico on a node that is not configured appropriately,
# label each Node so that the DaemonSet can run the Pods only on ready Nodes.
if [[ ${NETWORK_POLICY_PROVIDER:-} == "calico" ]]; then
	NON_MASTER_NODE_LABELS="${NON_MASTER_NODE_LABELS:+${NON_MASTER_NODE_LABELS},}projectcalico.org/ds-ready=true"
fi

# Enable metadata concealment by firewalling pod traffic to the metadata server
# and run a proxy daemonset on nodes.
ENABLE_METADATA_CONCEALMENT="${ENABLE_METADATA_CONCEALMENT:-true}" # true, false
if [[ ${ENABLE_METADATA_CONCEALMENT:-} == "true" ]]; then
  # Put the necessary label on the node so the daemonset gets scheduled.
  NODE_LABELS="${NODE_LABELS},beta.kubernetes.io/metadata-proxy-ready=true"
  # Add to the provider custom variables.
  PROVIDER_VARS="${PROVIDER_VARS:-} ENABLE_METADATA_CONCEALMENT METADATA_CONCEALMENT_NO_FIREWALL"
fi

# Optional: Enable node logging.
ENABLE_NODE_LOGGING="${KUBE_ENABLE_NODE_LOGGING:-true}"
LOGGING_DESTINATION="${KUBE_LOGGING_DESTINATION:-gcp}" # options: elasticsearch, gcp

# Optional: When set to true, Elasticsearch and Kibana will be setup as part of the cluster bring up.
ENABLE_CLUSTER_LOGGING="${KUBE_ENABLE_CLUSTER_LOGGING:-true}"
ELASTICSEARCH_LOGGING_REPLICAS=1

# Optional: Don't require https for registries in our local RFC1918 network
if [[ ${KUBE_ENABLE_INSECURE_REGISTRY:-false} == "true" ]]; then
  EXTRA_DOCKER_OPTS="${EXTRA_DOCKER_OPTS} --insecure-registry 10.0.0.0/8"
fi

if [[ ! -z "${NODE_ACCELERATORS}" ]]; then
    FEATURE_GATES="${FEATURE_GATES},DevicePlugins=true"
    if [[ "${NODE_ACCELERATORS}" =~ .*type=([a-zA-Z0-9-]+).* ]]; then
        NODE_LABELS="${NODE_LABELS},cloud.google.com/gke-accelerator=${BASH_REMATCH[1]}"
    fi
fi

# Optional: Install cluster DNS.
# Set CLUSTER_DNS_CORE_DNS to 'true' to install CoreDNS instead of kube-dns.
CLUSTER_DNS_CORE_DNS="${CLUSTER_DNS_CORE_DNS:-false}"
ENABLE_CLUSTER_DNS="${KUBE_ENABLE_CLUSTER_DNS:-true}"
DNS_SERVER_IP="10.0.0.10"
DNS_DOMAIN="cluster.local"

# Optional: Enable DNS horizontal autoscaler
ENABLE_DNS_HORIZONTAL_AUTOSCALER="${KUBE_ENABLE_DNS_HORIZONTAL_AUTOSCALER:-true}"

# Optional: Install Kubernetes UI
ENABLE_CLUSTER_UI="${KUBE_ENABLE_CLUSTER_UI:-true}"

# Optional: Install node problem detector.
#   none           - Not run node problem detector.
#   daemonset      - Run node problem detector as daemonset.
#   standalone     - Run node problem detector as standalone system daemon.
if [[ "${NODE_OS_DISTRIBUTION}" == "gci" ]]; then
  # Enable standalone mode by default for gci.
  ENABLE_NODE_PROBLEM_DETECTOR="${KUBE_ENABLE_NODE_PROBLEM_DETECTOR:-standalone}"
else
  ENABLE_NODE_PROBLEM_DETECTOR="${KUBE_ENABLE_NODE_PROBLEM_DETECTOR:-daemonset}"
fi
NODE_PROBLEM_DETECTOR_VERSION="${NODE_PROBLEM_DETECTOR_VERSION:-}"
NODE_PROBLEM_DETECTOR_TAR_HASH="${NODE_PROBLEM_DETECTOR_TAR_HASH:-}"

# Optional: Create autoscaler for cluster's nodes.
ENABLE_CLUSTER_AUTOSCALER="${KUBE_ENABLE_CLUSTER_AUTOSCALER:-false}"
if [[ "${ENABLE_CLUSTER_AUTOSCALER}" == "true" ]]; then
  AUTOSCALER_MIN_NODES="${KUBE_AUTOSCALER_MIN_NODES:-}"
  AUTOSCALER_MAX_NODES="${KUBE_AUTOSCALER_MAX_NODES:-}"
  AUTOSCALER_ENABLE_SCALE_DOWN="${KUBE_AUTOSCALER_ENABLE_SCALE_DOWN:-false}"
  AUTOSCALER_EXPANDER_CONFIG="${KUBE_AUTOSCALER_EXPANDER_CONFIG:---expander=price}"
fi

# Optional: Enable Rescheduler
ENABLE_RESCHEDULER="${KUBE_ENABLE_RESCHEDULER:-true}"

# Optional: Enable allocation of pod IPs using IP aliases.
#
# BETA FEATURE.
#
# IP_ALIAS_SIZE is the size of the podCIDR allocated to a node.
# IP_ALIAS_SUBNETWORK is the subnetwork to allocate from. If empty, a
#   new subnetwork will be created for the cluster.
ENABLE_IP_ALIASES=${KUBE_GCE_ENABLE_IP_ALIASES:-false}
NODE_IPAM_MODE=${KUBE_GCE_NODE_IPAM_MODE:-RangeAllocator}
if [ ${ENABLE_IP_ALIASES} = true ]; then
  # Size of ranges allocated to each node. gcloud current supports only /32 and /24.
  IP_ALIAS_SIZE=${KUBE_GCE_IP_ALIAS_SIZE:-/24}
  IP_ALIAS_SUBNETWORK=${KUBE_GCE_IP_ALIAS_SUBNETWORK:-${INSTANCE_PREFIX}-subnet-default}
  # If we're using custom network, use the subnet we already create for it as the one for ip-alias.
  # Note that this means SUBNETWORK would override KUBE_GCE_IP_ALIAS_SUBNETWORK in case of custom network.
  if [[ "${CREATE_CUSTOM_NETWORK}" == true ]]; then
    IP_ALIAS_SUBNETWORK="${SUBNETWORK:-IP_ALIAS_SUBNETWORK}"
  fi
  # Reserve the services IP space to avoid being allocated for other GCP resources.
  SERVICE_CLUSTER_IP_SUBNETWORK=${KUBE_GCE_SERVICE_CLUSTER_IP_SUBNETWORK:-${INSTANCE_PREFIX}-subnet-services}
  NODE_IPAM_MODE=${KUBE_GCE_NODE_IPAM_MODE:-CloudAllocator}
  SECONDARY_RANGE_NAME=${SECONDARY_RANGE_NAME:-}
  # Add to the provider custom variables.
  PROVIDER_VARS="${PROVIDER_VARS:-} ENABLE_IP_ALIASES"
  PROVIDER_VARS="${PROVIDER_VARS:-} NODE_IPAM_MODE"
  PROVIDER_VARS="${PROVIDER_VARS:-} SECONDARY_RANGE_NAME"
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

if [[ -z "${KUBE_ADMISSION_CONTROL:-}" ]]; then
  ADMISSION_CONTROL="Initializers,NamespaceLifecycle,LimitRanger,ServiceAccount,PersistentVolumeLabel,PodPreset,DefaultStorageClass,DefaultTolerationSeconds,NodeRestriction,Priority,StorageObjectInUseProtection"
  if [[ "${ENABLE_POD_SECURITY_POLICY:-}" == "true" ]]; then
    ADMISSION_CONTROL="${ADMISSION_CONTROL},PodSecurityPolicy"
  fi
  # ResourceQuota must come last, or a creation is recorded, but the pod may be forbidden.
  ADMISSION_CONTROL="${ADMISSION_CONTROL},MutatingAdmissionWebhook,ValidatingAdmissionWebhook,ResourceQuota"
else
  ADMISSION_CONTROL=${KUBE_ADMISSION_CONTROL}
fi

# Optional: if set to true kube-up will automatically check for existing resources and clean them up.
KUBE_UP_AUTOMATIC_CLEANUP=${KUBE_UP_AUTOMATIC_CLEANUP:-false}

# Optional: setting it to true denotes this is a testing cluster,
# so that we can use pulled kubernetes binaries, even if binaries
# are pre-installed in the image. Note that currently this logic
# is only supported in trusty or GCI.
TEST_CLUSTER="${TEST_CLUSTER:-true}"

# Storage backend. 'etcd2' and 'etcd3' are supported.
STORAGE_BACKEND=${STORAGE_BACKEND:-}
# Storage media type: application/json and application/vnd.kubernetes.protobuf are supported.
STORAGE_MEDIA_TYPE=${STORAGE_MEDIA_TYPE:-}

NETWORK_PROVIDER="${NETWORK_PROVIDER:-kubenet}" # none, kubenet

# Network Policy plugin specific settings.
NETWORK_POLICY_PROVIDER="${NETWORK_POLICY_PROVIDER:-none}" # calico

NON_MASQUERADE_CIDR="0.0.0.0/0"

# How should the kubelet configure hairpin mode?
HAIRPIN_MODE="${HAIRPIN_MODE:-promiscuous-bridge}" # promiscuous-bridge, hairpin-veth, none

# Optional: if set to true, kube-up will configure the cluster to run e2e tests.
E2E_STORAGE_TEST_ENVIRONMENT=${KUBE_E2E_STORAGE_TEST_ENVIRONMENT:-false}

# Optional: if set to true, a image puller is deployed. Only for use in e2e clusters.
# TODO: Pipe this through GKE e2e clusters once we know it helps.
PREPULL_E2E_IMAGES="${PREPULL_E2E_IMAGES:-true}"

# Evict pods whenever compute resource availability on the nodes gets below a threshold.
EVICTION_HARD="${EVICTION_HARD:-memory.available<250Mi,nodefs.available<10%,nodefs.inodesFree<5%}"

# Optional: custom scheduling algorithm
SCHEDULING_ALGORITHM_PROVIDER="${SCHEDULING_ALGORITHM_PROVIDER:-}"

# Optional: install a default StorageClass
ENABLE_DEFAULT_STORAGE_CLASS="${ENABLE_DEFAULT_STORAGE_CLASS:-true}"

# Optional: Enable legacy ABAC policy that makes all service accounts superusers.
# Disabling this by default in tests ensures default RBAC policies are sufficient from 1.6+
# Upgrade test jobs that go from a version < 1.6 to a version >= 1.6 should override this to be true.
ENABLE_LEGACY_ABAC="${ENABLE_LEGACY_ABAC:-false}" # true, false

# Enable a simple "AdvancedAuditing" setup for testing.
ENABLE_APISERVER_ADVANCED_AUDIT="${ENABLE_APISERVER_ADVANCED_AUDIT:-true}" # true, false
ADVANCED_AUDIT_LOG_MODE="${ADVANCED_AUDIT_LOG_MODE:-batch}" # batch, blocking

if [[ "${ENABLE_APISERVER_BASIC_AUDIT:-}" == "true" ]]; then
  echo "Warning: Basic audit logging is deprecated and will be removed. Please use advanced auditing instead."
fi

ENABLE_BIG_CLUSTER_SUBNETS="${ENABLE_BIG_CLUSTER_SUBNETS:-false}"

if [[ -n "${LOGROTATE_FILES_MAX_COUNT:-}" ]]; then
  PROVIDER_VARS="${PROVIDER_VARS:-} LOGROTATE_FILES_MAX_COUNT"
fi
if [[ -n "${LOGROTATE_MAX_SIZE:-}" ]]; then
  PROVIDER_VARS="${PROVIDER_VARS:-} LOGROTATE_MAX_SIZE"
fi

# Fluentd requirements
FLUENTD_GCP_VERSION="${FLUENTD_GCP_VERSION:-0.2-1.5.30-1-k8s}"
FLUENTD_GCP_MEMORY_LIMIT="${FLUENTD_GCP_MEMORY_LIMIT:-}"
FLUENTD_GCP_CPU_REQUEST="${FLUENTD_GCP_CPU_REQUEST:-}"
FLUENTD_GCP_MEMORY_REQUEST="${FLUENTD_GCP_MEMORY_REQUEST:-}"

# Heapster requirements
HEAPSTER_GCP_BASE_MEMORY="${HEAPSTER_GCP_BASE_MEMORY:-140Mi}"
HEAPSTER_GCP_MEMORY_PER_NODE="${HEAPSTER_GCP_MEMORY_PER_NODE:-4}"
HEAPSTER_GCP_BASE_CPU="${HEAPSTER_GCP_BASE_CPU:-80m}"
HEAPSTER_GCP_CPU_PER_NODE="${HEAPSTER_GCP_CPU_PER_NODE:-0.5}"

# Adding to PROVIDER_VARS, since this is GCP-specific.
PROVIDER_VARS="${PROVIDER_VARS:-} FLUENTD_GCP_VERSION FLUENTD_GCP_MEMORY_LIMIT FLUENTD_GCP_CPU_REQUEST FLUENTD_GCP_MEMORY_REQUEST HEAPSTER_GCP_BASE_MEMORY HEAPSTER_GCP_MEMORY_PER_NODE HEAPSTER_GCP_BASE_CPU HEAPSTER_GCP_CPU_PER_NODE CUSTOM_KUBE_DASHBOARD_BANNER LOGGING_STACKDRIVER_RESOURCE_TYPES STACKDRIVER_METADATA_AGENT_URL"

# prometheus-to-sd configuration
PROMETHEUS_TO_SD_ENDPOINT="${PROMETHEUS_TO_SD_ENDPOINT:-https://monitoring.googleapis.com/}"
PROMETHEUS_TO_SD_PREFIX="${PROMETHEUS_TO_SD_PREFIX:-custom.googleapis.com}"
ENABLE_PROMETHEUS_TO_SD="${ENABLE_PROMETHEUS_TO_SD:-true}"

# TODO(#51292): Make kube-proxy Daemonset default and remove the configuration here.
# Optional: [Experiment Only] Run kube-proxy as a DaemonSet if set to true, run as static pods otherwise.
KUBE_PROXY_DAEMONSET="${KUBE_PROXY_DAEMONSET:-false}" # true, false

# Optional: Change the kube-proxy implementation. Choices are [iptables, ipvs].
KUBE_PROXY_MODE="${KUBE_PROXY_MODE:-iptables}"

# Optional: duration of cluster signed certificates.
CLUSTER_SIGNING_DURATION="${CLUSTER_SIGNING_DURATION:-}"

# Optional: enable pod priority
ENABLE_POD_PRIORITY="${ENABLE_POD_PRIORITY:-}"
if [[ "${ENABLE_POD_PRIORITY}" == "true" ]]; then
    FEATURE_GATES="${FEATURE_GATES},PodPriority=true"
fi

# Optional: enable certificate rotation of the kubelet certificates.
ROTATE_CERTIFICATES="${ROTATE_CERTIFICATES:-}"

# The number of services that are allowed to sync concurrently. Will be passed
# into kube-controller-manager via `--concurrent-service-syncs`
CONCURRENT_SERVICE_SYNCS="${CONCURRENT_SERVICE_SYNCS:-}"

if [[ "${ENABLE_TOKENREQUEST:-}" == "true" ]]; then
  FEATURE_GATES="${FEATURE_GATES},TokenRequest=true"
  SERVICEACCOUNT_ISSUER="https://kubernetes.io/${CLUSTER_NAME}"
  SERVICEACCOUNT_API_AUDIENCES="https://kubernetes.default.svc"
fi
