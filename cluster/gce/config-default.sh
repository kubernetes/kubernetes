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

# Historically fluentd was a manifest pod and then was migrated to DaemonSet.
# To avoid situation during cluster upgrade when there are two instances
# of fluentd running on a node, kubelet need to mark node on which
# fluentd is not running as a manifest pod with appropriate label.
# TODO(piosz): remove this in 1.8
NODE_LABELS="${KUBE_NODE_LABELS:-beta.kubernetes.io/fluentd-ds-ready=true}"

# An extension to local SSDs allowing users to specify block/fs and SCSI/NVMe devices
# Format of this variable will be "#,scsi/nvme,block/fs" you can specify multiple
# configurations by separating them by a semi-colon ex. "2,scsi,fs;1,nvme,block"
# is a request for 2 SCSI formatted and mounted SSDs and 1 NVMe block device SSD.
NODE_LOCAL_SSDS_EXT=${NODE_LOCAL_SSDS_EXT:-}
# Accelerators to be attached to each node. Format "type=<accelerator-type>,count=<accelerator-count>"
# More information on available GPUs here - https://cloud.google.com/compute/docs/gpus/
NODE_ACCELERATORS=${NODE_ACCELERATORS:-""}
REGISTER_MASTER_KUBELET=${REGISTER_MASTER:-true}
PREEMPTIBLE_NODE=${PREEMPTIBLE_NODE:-false}
PREEMPTIBLE_MASTER=${PREEMPTIBLE_MASTER:-false}
KUBE_DELETE_NODES=${KUBE_DELETE_NODES:-true}
KUBE_DELETE_NETWORK=${KUBE_DELETE_NETWORK:-} # default value calculated below
CREATE_CUSTOM_NETWORK=${CREATE_CUSTOM_NETWORK:-false}
MIG_WAIT_UNTIL_STABLE_TIMEOUT=${MIG_WAIT_UNTIL_STABLE_TIMEOUT:-1800}

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
GCI_VERSION=${KUBE_GCI_VERSION:-cos-stable-65-10323-64-0}
MASTER_IMAGE=${KUBE_GCE_MASTER_IMAGE:-}
MASTER_IMAGE_PROJECT=${KUBE_GCE_MASTER_PROJECT:-cos-cloud}
NODE_IMAGE=${KUBE_GCE_NODE_IMAGE:-${GCI_VERSION}}
NODE_IMAGE_PROJECT=${KUBE_GCE_NODE_PROJECT:-cos-cloud}
NODE_SERVICE_ACCOUNT=${KUBE_GCE_NODE_SERVICE_ACCOUNT:-default}
CONTAINER_RUNTIME=${KUBE_CONTAINER_RUNTIME:-docker}
CONTAINER_RUNTIME_ENDPOINT=${KUBE_CONTAINER_RUNTIME_ENDPOINT:-}
CONTAINER_RUNTIME_NAME=${KUBE_CONTAINER_RUNTIME_NAME:-}
LOAD_IMAGE_COMMAND=${KUBE_LOAD_IMAGE_COMMAND:-}
# MASTER_EXTRA_METADATA is the extra instance metadata on master instance separated by commas.
MASTER_EXTRA_METADATA=${KUBE_MASTER_EXTRA_METADATA:-${KUBE_EXTRA_METADATA:-}}
# MASTER_EXTRA_METADATA is the extra instance metadata on node instance separated by commas.
NODE_EXTRA_METADATA=${KUBE_NODE_EXTRA_METADATA:-${KUBE_EXTRA_METADATA:-}}
# KUBELET_TEST_ARGS are extra arguments passed to kubelet.
KUBELET_TEST_ARGS=${KUBE_KUBELET_EXTRA_ARGS:-}

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
AGGREGATOR_MASTER_NAME="${INSTANCE_PREFIX}-aggregator"
INITIAL_ETCD_CLUSTER="${MASTER_NAME}"
MASTER_TAG="${INSTANCE_PREFIX}-master"
NODE_TAG="${INSTANCE_PREFIX}-minion"

CLUSTER_IP_RANGE="${CLUSTER_IP_RANGE:-$(get-cluster-ip-range)}"
MASTER_IP_RANGE="${MASTER_IP_RANGE:-10.246.0.0/24}"
# NODE_IP_RANGE is used when ENABLE_IP_ALIASES=true or CREATE_CUSTOM_NETWORK=true.
# It is the primary range in the subnet and is the range used for node instance IPs.
NODE_IP_RANGE="$(get-node-ip-range)"

# NOTE: Avoid giving nodes empty scopes, because kubelet needs a service account
# in order to initialize properly.
NODE_SCOPES="${NODE_SCOPES:-monitoring,logging-write,storage-ro}"

# Extra docker options for nodes.
EXTRA_DOCKER_OPTS="${EXTRA_DOCKER_OPTS:-}"

VOLUME_PLUGIN_DIR="${VOLUME_PLUGIN_DIR:-/home/kubernetes/flexvolume}"

SERVICE_CLUSTER_IP_RANGE="${SERVICE_CLUSTER_IP_RANGE:-10.0.0.0/16}"  # formerly PORTAL_NET
ALLOCATE_NODE_CIDRS=true

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
ENABLE_CLUSTER_MONITORING="${KUBE_ENABLE_CLUSTER_MONITORING:-standalone}"

# Optional: Enable deploying separate prometheus stack for monitoring kubernetes cluster
ENABLE_PROMETHEUS_MONITORING="${KUBE_ENABLE_PROMETHEUS_MONITORING:-false}"

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

# NON_MASTER_NODE_LABELS are labels will only be applied on non-master nodes.
NON_MASTER_NODE_LABELS="${KUBE_NON_MASTER_NODE_LABELS:-}"

if [[ "${PREEMPTIBLE_MASTER}" == "true" ]]; then
    NODE_LABELS="${NODE_LABELS},cloud.google.com/gke-preemptible=true"
elif [[ "${PREEMPTIBLE_NODE}" == "true" ]]; then
    NON_MASTER_NODE_LABELS="${NON_MASTER_NODE_LABELS},cloud.google.com/gke-preemptible=true"
fi

# To avoid running Calico on a node that is not configured appropriately,
# label each Node so that the DaemonSet can run the Pods only on ready Nodes.
if [[ ${NETWORK_POLICY_PROVIDER:-} == "calico" ]]; then
	NON_MASTER_NODE_LABELS="${NON_MASTER_NODE_LABELS:+${NON_MASTER_NODE_LABELS},}projectcalico.org/ds-ready=true"
fi

# Optional: Enable netd.
ENABLE_NETD="${KUBE_ENABLE_NETD:-false}"
CUSTOM_NETD_YAML="${KUBE_CUSTOM_NETD_YAML:-}"
CUSTOM_CALICO_NODE_DAEMONSET_YAML="${KUBE_CUSTOM_CALICO_NODE_DAEMONSET_YAML:-}"
CUSTOM_TYPHA_DEPLOYMENT_YAML="${KUBE_CUSTOM_TYPHA_DEPLOYMENT_YAML:-}"

# To avoid running netd on a node that is not configured appropriately,
# label each Node so that the DaemonSet can run the Pods only on ready Nodes.
if [[ ${ENABLE_NETD:-} == "true" ]]; then
	NON_MASTER_NODE_LABELS="${NON_MASTER_NODE_LABELS:+${NON_MASTER_NODE_LABELS},}cloud.google.com/gke-netd-ready=true"
fi

ENABLE_NODELOCAL_DNS="${KUBE_ENABLE_NODELOCAL_DNS:-false}"
LOCAL_DNS_IP="${KUBE_LOCAL_DNS_IP:-169.254.20.10}"

# Enable metadata concealment by firewalling pod traffic to the metadata server
# and run a proxy daemonset on nodes.
#
# TODO(#8867) Enable by default.
ENABLE_METADATA_CONCEALMENT="${ENABLE_METADATA_CONCEALMENT:-false}" # true, false
METADATA_CONCEALMENT_NO_FIREWALL="${METADATA_CONCEALMENT_NO_FIREWALL:-false}" # true, false
if [[ ${ENABLE_METADATA_CONCEALMENT:-} == "true" ]]; then
  # Put the necessary label on the node so the daemonset gets scheduled.
  NODE_LABELS="${NODE_LABELS},beta.kubernetes.io/metadata-proxy-ready=true"
  # Add to the provider custom variables.
  PROVIDER_VARS="${PROVIDER_VARS:-} ENABLE_METADATA_CONCEALMENT METADATA_CONCEALMENT_NO_FIREWALL"
fi


# Enable AESGCM encryption of secrets by default.
ENCRYPTION_PROVIDER_CONFIG="${ENCRYPTION_PROVIDER_CONFIG:-}"
if [[ -z "${ENCRYPTION_PROVIDER_CONFIG}" ]]; then
    ENCRYPTION_PROVIDER_CONFIG=$(cat << EOM | base64 | tr -d '\r\n'
kind: EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
resources:
  - resources:
    - secrets
    providers:
    - aesgcm:
        keys:
        - name: key1
          secret: $(dd if=/dev/urandom iflag=fullblock bs=32 count=1 2>/dev/null | base64 | tr -d '\r\n')
EOM
)
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

# Optional: customize runtime config
RUNTIME_CONFIG="${KUBE_RUNTIME_CONFIG:-}"

if [[ "${KUBE_FEATURE_GATES:-}" == "AllAlpha=true" ]]; then
  RUNTIME_CONFIG="${KUBE_RUNTIME_CONFIG:-api/all=true}"
fi

# Optional: set feature gates
FEATURE_GATES="${KUBE_FEATURE_GATES:-ExperimentalCriticalPodAnnotation=true}"

if [[ ! -z "${NODE_ACCELERATORS}" ]]; then
    FEATURE_GATES="${FEATURE_GATES},DevicePlugins=true"
    if [[ "${NODE_ACCELERATORS}" =~ .*type=([a-zA-Z0-9-]+).* ]]; then
        NON_MASTER_NODE_LABELS="${NON_MASTER_NODE_LABELS},cloud.google.com/gke-accelerator=${BASH_REMATCH[1]}"
    fi
fi

# Optional: Install cluster DNS.
# Set CLUSTER_DNS_CORE_DNS to 'false' to install kube-dns instead of CoreDNS.
CLUSTER_DNS_CORE_DNS="${CLUSTER_DNS_CORE_DNS:-true}"
ENABLE_CLUSTER_DNS="${KUBE_ENABLE_CLUSTER_DNS:-true}"
DNS_SERVER_IP="${KUBE_DNS_SERVER_IP:-10.0.0.10}"
DNS_DOMAIN="${KUBE_DNS_DOMAIN:-cluster.local}"

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
  AUTOSCALER_ENABLE_SCALE_DOWN="${KUBE_AUTOSCALER_ENABLE_SCALE_DOWN:-true}"
  AUTOSCALER_EXPANDER_CONFIG="${KUBE_AUTOSCALER_EXPANDER_CONFIG:---expander=price}"
fi

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
  # Number of Pods that can run on this node.
  MAX_PODS_PER_NODE=${MAX_PODS_PER_NODE:-110}
  # Size of ranges allocated to each node.
  IP_ALIAS_SIZE="/$(get-alias-range-size ${MAX_PODS_PER_NODE})"
  IP_ALIAS_SUBNETWORK=${KUBE_GCE_IP_ALIAS_SUBNETWORK:-${INSTANCE_PREFIX}-subnet-default}
  # If we're using custom network, use the subnet we already create for it as the one for ip-alias.
  # Note that this means SUBNETWORK would override KUBE_GCE_IP_ALIAS_SUBNETWORK in case of custom network.
  if [[ "${CREATE_CUSTOM_NETWORK}" == true ]]; then
    IP_ALIAS_SUBNETWORK="${SUBNETWORK}"
  fi
  # Reserve the services IP space to avoid being allocated for other GCP resources.
  SERVICE_CLUSTER_IP_SUBNETWORK=${KUBE_GCE_SERVICE_CLUSTER_IP_SUBNETWORK:-${INSTANCE_PREFIX}-subnet-services}
  NODE_IPAM_MODE=${KUBE_GCE_NODE_IPAM_MODE:-CloudAllocator}
  SECONDARY_RANGE_NAME=${SECONDARY_RANGE_NAME:-}
  # Add to the provider custom variables.
  PROVIDER_VARS="${PROVIDER_VARS:-} ENABLE_IP_ALIASES"
  PROVIDER_VARS="${PROVIDER_VARS:-} NODE_IPAM_MODE"
  PROVIDER_VARS="${PROVIDER_VARS:-} SECONDARY_RANGE_NAME"
elif [[ -n "${MAX_PODS_PER_NODE:-}" ]]; then
  # Should not have MAX_PODS_PER_NODE set for route-based clusters.
  echo -e "${color_red}Cannot set MAX_PODS_PER_NODE for route-based projects for ${PROJECT}." >&2
  exit 1
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

# Admission Controllers to invoke prior to persisting objects in cluster
ADMISSION_CONTROL=NamespaceLifecycle,LimitRanger,ServiceAccount,PersistentVolumeLabel,DefaultStorageClass,PersistentVolumeClaimResize,DefaultTolerationSeconds,NodeRestriction,Priority,StorageObjectInUseProtection

if [[ "${ENABLE_POD_SECURITY_POLICY:-}" == "true" ]]; then
  ADMISSION_CONTROL="${ADMISSION_CONTROL},PodSecurityPolicy"
fi

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

NON_MASQUERADE_CIDR="0.0.0.0/0"

# How should the kubelet configure hairpin mode?
HAIRPIN_MODE="${HAIRPIN_MODE:-hairpin-veth}" # promiscuous-bridge, hairpin-veth, none
# Optional: if set to true, kube-up will configure the cluster to run e2e tests.
E2E_STORAGE_TEST_ENVIRONMENT="${KUBE_E2E_STORAGE_TEST_ENVIRONMENT:-false}"

# Evict pods whenever compute resource availability on the nodes gets below a threshold.
EVICTION_HARD="${EVICTION_HARD:-memory.available<250Mi,nodefs.available<10%,nodefs.inodesFree<5%}"

# Optional: custom scheduling algorithm
SCHEDULING_ALGORITHM_PROVIDER="${SCHEDULING_ALGORITHM_PROVIDER:-}"

# Optional: install a default StorageClass
ENABLE_DEFAULT_STORAGE_CLASS="${ENABLE_DEFAULT_STORAGE_CLASS:-true}"

# Optional: Enable legacy ABAC policy that makes all service accounts superusers.
ENABLE_LEGACY_ABAC="${ENABLE_LEGACY_ABAC:-false}" # true, false

# Indicates if the values (i.e. KUBE_USER and KUBE_PASSWORD for basic
# authentication) in metadata should be treated as canonical, and therefore disk
# copies ought to be recreated/clobbered.
METADATA_CLOBBERS_CONFIG="${METADATA_CLOBBERS_CONFIG:-false}"

ENABLE_BIG_CLUSTER_SUBNETS="${ENABLE_BIG_CLUSTER_SUBNETS:-false}"

if [[ -n "${LOGROTATE_FILES_MAX_COUNT:-}" ]]; then
  PROVIDER_VARS="${PROVIDER_VARS:-} LOGROTATE_FILES_MAX_COUNT"
fi
if [[ -n "${LOGROTATE_MAX_SIZE:-}" ]]; then
  PROVIDER_VARS="${PROVIDER_VARS:-} LOGROTATE_MAX_SIZE"
fi

# Fluentd requirements
# YAML exists to trigger a configuration refresh when changes are made.
FLUENTD_GCP_YAML_VERSION="v3.2.0"
FLUENTD_GCP_VERSION="${FLUENTD_GCP_VERSION:-0.6-1.6.0-1}"
FLUENTD_GCP_MEMORY_LIMIT="${FLUENTD_GCP_MEMORY_LIMIT:-}"
FLUENTD_GCP_CPU_REQUEST="${FLUENTD_GCP_CPU_REQUEST:-}"
FLUENTD_GCP_MEMORY_REQUEST="${FLUENTD_GCP_MEMORY_REQUEST:-}"

# Heapster requirements
HEAPSTER_GCP_BASE_MEMORY="${HEAPSTER_GCP_BASE_MEMORY:-140Mi}"
HEAPSTER_GCP_MEMORY_PER_NODE="${HEAPSTER_GCP_MEMORY_PER_NODE:-4}"
HEAPSTER_GCP_BASE_CPU="${HEAPSTER_GCP_BASE_CPU:-80m}"
HEAPSTER_GCP_CPU_PER_NODE="${HEAPSTER_GCP_CPU_PER_NODE:-0.5}"

# Optional: custom system banner for dashboard addon
CUSTOM_KUBE_DASHBOARD_BANNER="${CUSTOM_KUBE_DASHBOARD_BANNER:-}"

# Default Stackdriver resources version exported by Fluentd-gcp addon
LOGGING_STACKDRIVER_RESOURCE_TYPES="${LOGGING_STACKDRIVER_RESOURCE_TYPES:-old}"

# Adding to PROVIDER_VARS, since this is GCP-specific.
PROVIDER_VARS="${PROVIDER_VARS:-} FLUENTD_GCP_YAML_VERSION FLUENTD_GCP_VERSION FLUENTD_GCP_MEMORY_LIMIT FLUENTD_GCP_CPU_REQUEST FLUENTD_GCP_MEMORY_REQUEST HEAPSTER_GCP_BASE_MEMORY HEAPSTER_GCP_MEMORY_PER_NODE HEAPSTER_GCP_BASE_CPU HEAPSTER_GCP_CPU_PER_NODE CUSTOM_KUBE_DASHBOARD_BANNER LOGGING_STACKDRIVER_RESOURCE_TYPES"

# Fluentd configuration for node-journal
ENABLE_NODE_JOURNAL="${ENABLE_NODE_JOURNAL:-false}"

# prometheus-to-sd configuration
PROMETHEUS_TO_SD_ENDPOINT="${PROMETHEUS_TO_SD_ENDPOINT:-https://monitoring.googleapis.com/}"
PROMETHEUS_TO_SD_PREFIX="${PROMETHEUS_TO_SD_PREFIX:-custom.googleapis.com}"
ENABLE_PROMETHEUS_TO_SD="${ENABLE_PROMETHEUS_TO_SD:-false}"

# TODO(#51292): Make kube-proxy Daemonset default and remove the configuration here.
# Optional: [Experiment Only] Run kube-proxy as a DaemonSet if set to true, run as static pods otherwise.
KUBE_PROXY_DAEMONSET="${KUBE_PROXY_DAEMONSET:-false}" # true, false

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

SERVICEACCOUNT_ISSUER="https://kubernetes.io/${CLUSTER_NAME}"

# Optional: Enable Node termination Handler for Preemptible and GPU VMs.
# https://github.com/GoogleCloudPlatform/k8s-node-termination-handler
ENABLE_NODE_TERMINATION_HANDLER="${ENABLE_NODE_TERMINATION_HANDLER:-false}"
# Override default Node Termination Handler Image
if [[ "${NODE_TERMINATION_HANDLER_IMAGE:-}" ]]; then
  PROVIDER_VARS="${PROVIDER_VARS:-} NODE_TERMINATION_HANDLER_IMAGE"
fi
