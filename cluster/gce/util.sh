#!/bin/bash

# Copyright 2017 The Kubernetes Authors.
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

# A library of helper functions and constant for the local config.

# Use the config file specified in $KUBE_CONFIG_FILE, or default to
# config-default.sh.
readonly GCE_MAX_LOCAL_SSD=8

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/cluster/gce/${KUBE_CONFIG_FILE-"config-default.sh"}"
source "${KUBE_ROOT}/cluster/common.sh"
source "${KUBE_ROOT}/hack/lib/util.sh"

if [[ "${NODE_OS_DISTRIBUTION}" == "gci" || "${NODE_OS_DISTRIBUTION}" == "ubuntu" || "${NODE_OS_DISTRIBUTION}" == "custom" ]]; then
  source "${KUBE_ROOT}/cluster/gce/${NODE_OS_DISTRIBUTION}/node-helper.sh"
else
  echo "Cannot operate on cluster using node os distro: ${NODE_OS_DISTRIBUTION}" >&2
  exit 1
fi

if [[ "${MASTER_OS_DISTRIBUTION}" == "trusty" || "${MASTER_OS_DISTRIBUTION}" == "gci" || "${MASTER_OS_DISTRIBUTION}" == "ubuntu" ]]; then
  source "${KUBE_ROOT}/cluster/gce/${MASTER_OS_DISTRIBUTION}/master-helper.sh"
else
  echo "Cannot operate on cluster using master os distro: ${MASTER_OS_DISTRIBUTION}" >&2
  exit 1
fi

if [[ ${NODE_LOCAL_SSDS:-} -ge 1 ]] && [[ ! -z ${NODE_LOCAL_SSDS_EXT:-} ]] ; then
  echo -e "${color_red}Local SSD: Only one of NODE_LOCAL_SSDS and NODE_LOCAL_SSDS_EXT can be specified at once${color_norm}" >&2
  exit 2
fi

if [[ "${MASTER_OS_DISTRIBUTION}" == "gci" ]]; then
    DEFAULT_GCI_PROJECT=google-containers
    if [[ "${GCI_VERSION}" == "cos"* ]]; then
        DEFAULT_GCI_PROJECT=cos-cloud
    fi
    MASTER_IMAGE_PROJECT=${KUBE_GCE_MASTER_PROJECT:-${DEFAULT_GCI_PROJECT}}
    # If the master image is not set, we use the latest GCI image.
    # Otherwise, we respect whatever is set by the user.
    MASTER_IMAGE=${KUBE_GCE_MASTER_IMAGE:-${GCI_VERSION}}
fi

# Sets node image based on the specified os distro. Currently this function only
# supports gci and debian.
function set-node-image() {
    if [[ "${NODE_OS_DISTRIBUTION}" == "gci" ]]; then
        DEFAULT_GCI_PROJECT=google-containers
        if [[ "${GCI_VERSION}" == "cos"* ]]; then
            DEFAULT_GCI_PROJECT=cos-cloud
        fi

        # If the node image is not set, we use the latest GCI image.
        # Otherwise, we respect whatever is set by the user.
        NODE_IMAGE=${KUBE_GCE_NODE_IMAGE:-${GCI_VERSION}}
        NODE_IMAGE_PROJECT=${KUBE_GCE_NODE_PROJECT:-${DEFAULT_GCI_PROJECT}}
    fi
}

set-node-image

# Verfiy cluster autoscaler configuration.
if [[ "${ENABLE_CLUSTER_AUTOSCALER}" == "true" ]]; then
  if [[ -z $AUTOSCALER_MIN_NODES ]]; then
    echo "AUTOSCALER_MIN_NODES not set."
    exit 1
  fi
  if [[ -z $AUTOSCALER_MAX_NODES ]]; then
    echo "AUTOSCALER_MAX_NODES not set."
    exit 1
  fi
fi

NODE_INSTANCE_PREFIX=${NODE_INSTANCE_PREFIX:-"${INSTANCE_PREFIX}-minion"}

NODE_TAGS="${NODE_TAG}"

ALLOCATE_NODE_CIDRS=true
PREEXISTING_NETWORK=false
PREEXISTING_NETWORK_MODE=""

KUBE_PROMPT_FOR_UPDATE=${KUBE_PROMPT_FOR_UPDATE:-"n"}
# How long (in seconds) to wait for cluster initialization.
KUBE_CLUSTER_INITIALIZATION_TIMEOUT=${KUBE_CLUSTER_INITIALIZATION_TIMEOUT:-300}

function join_csv() {
  local IFS=','; echo "$*";
}

# This function returns the first string before the comma
function split_csv() {
  echo "$*" | cut -d',' -f1
}

# Verify prereqs
function verify-prereqs() {
  local cmd
  for cmd in gcloud gsutil; do
    if ! which "${cmd}" >/dev/null; then
      local resp="n"
      if [[ "${KUBE_PROMPT_FOR_UPDATE}" == "y" ]]; then
        echo "Can't find ${cmd} in PATH.  Do you wish to install the Google Cloud SDK? [Y/n]"
        read resp
      fi
      if [[ "${resp}" != "n" && "${resp}" != "N" ]]; then
        curl https://sdk.cloud.google.com | bash
      fi
      if ! which "${cmd}" >/dev/null; then
        echo "Can't find ${cmd} in PATH, please fix and retry. The Google Cloud " >&2
        echo "SDK can be downloaded from https://cloud.google.com/sdk/." >&2
        exit 1
      fi
    fi
  done
  update-or-verify-gcloud
}

# Use the gcloud defaults to find the project.  If it is already set in the
# environment then go with that.
#
# Vars set:
#   PROJECT
#   NETWORK_PROJECT
#   PROJECT_REPORTED
function detect-project() {
  if [[ -z "${PROJECT-}" ]]; then
    PROJECT=$(gcloud config list project --format 'value(core.project)')
  fi

  NETWORK_PROJECT=${NETWORK_PROJECT:-${PROJECT}}

  if [[ -z "${PROJECT-}" ]]; then
    echo "Could not detect Google Cloud Platform project.  Set the default project using " >&2
    echo "'gcloud config set project <PROJECT>'" >&2
    exit 1
  fi
  if [[ -z "${PROJECT_REPORTED-}" ]]; then
    echo "Project: ${PROJECT}" >&2
    echo "Network Project: ${NETWORK_PROJECT}" >&2
    echo "Zone: ${ZONE}" >&2
    PROJECT_REPORTED=true
  fi
}

# Use gsutil to get the md5 hash for a particular tar
function gsutil_get_tar_md5() {
  # location_tar could be local or in the cloud
  # local tar_location example ./_output/release-tars/kubernetes-server-linux-amd64.tar.gz
  # cloud tar_location example gs://kubernetes-staging-PROJECT/kubernetes-devel/kubernetes-server-linux-amd64.tar.gz
  local -r tar_location=$1
  #parse the output and return the md5 hash
  #the sed command at the end removes whitespace
  local -r tar_md5=$(gsutil hash -h -m ${tar_location} 2>/dev/null | grep "Hash (md5):" | awk -F ':' '{print $2}' | sed 's/^[[:space:]]*//g')
  echo "${tar_md5}"
}

# Copy a release tar and its accompanying hash.
function copy-to-staging() {
  local -r staging_path=$1
  local -r gs_url=$2
  local -r tar=$3
  local -r hash=$4
  local -r basename_tar=$(basename ${tar})

  #check whether this tar alread exists and has the same hash
  #if it matches, then don't bother uploading it again

  #remote_tar_md5 checks the remote location for the existing tarball and its md5
  #staging_path example gs://kubernetes-staging-PROJECT/kubernetes-devel
  #basename_tar example kubernetes-server-linux-amd64.tar.gz
  local -r remote_tar_md5=$(gsutil_get_tar_md5 "${staging_path}/${basename_tar}")
  if [[ -n ${remote_tar_md5} ]]; then
    #local_tar_md5 checks the remote location for the existing tarball and its md5 hash
    #tar example ./_output/release-tars/kubernetes-server-linux-amd64.tar.gz
    local -r local_tar_md5=$(gsutil_get_tar_md5 "${tar}")
    if [[ "${remote_tar_md5}" == "${local_tar_md5}" ]]; then
      echo "+++ ${basename_tar} uploaded earlier, cloud and local file md5 match (md5 = ${local_tar_md5})"
      return 0
    fi
  fi

  echo "${hash}" > "${tar}.sha1"
  gsutil -m -q -h "Cache-Control:private, max-age=0" cp "${tar}" "${tar}.sha1" "${staging_path}"
  gsutil -m acl ch -g all:R "${gs_url}" "${gs_url}.sha1" >/dev/null 2>&1
  echo "+++ ${basename_tar} uploaded (sha1 = ${hash})"
}


# Given the cluster zone, return the list of regional GCS release
# bucket suffixes for the release in preference order. GCS doesn't
# give us an API for this, so we hardcode it.
#
# Assumed vars:
#   RELEASE_REGION_FALLBACK
#   REGIONAL_KUBE_ADDONS
#   ZONE
# Vars set:
#   PREFERRED_REGION
function set-preferred-region() {
  case ${ZONE} in
    asia-*)
      PREFERRED_REGION=("asia" "us" "eu")
      ;;
    europe-*)
      PREFERRED_REGION=("eu" "us" "asia")
      ;;
    *)
      PREFERRED_REGION=("us" "eu" "asia")
      ;;
  esac

  if [[ "${RELEASE_REGION_FALLBACK}" != "true" ]]; then
    PREFERRED_REGION=( "${PREFERRED_REGION[0]}" )
  fi
}

# Take the local tar files and upload them to Google Storage.  They will then be
# downloaded by the master as part of the start up script for the master.
#
# Assumed vars:
#   PROJECT
#   SERVER_BINARY_TAR
#   KUBE_MANIFESTS_TAR
#   ZONE
# Vars set:
#   SERVER_BINARY_TAR_URL
#   SERVER_BINARY_TAR_HASH
#   KUBE_MANIFESTS_TAR_URL
#   KUBE_MANIFESTS_TAR_HASH
function upload-server-tars() {
  SERVER_BINARY_TAR_URL=
  SERVER_BINARY_TAR_HASH=
  KUBE_MANIFESTS_TAR_URL=
  KUBE_MANIFESTS_TAR_HASH=

  local project_hash
  if which md5 > /dev/null 2>&1; then
    project_hash=$(md5 -q -s "$PROJECT")
  else
    project_hash=$(echo -n "$PROJECT" | md5sum | awk '{ print $1 }')
  fi

  # This requires 1 million projects before the probability of collision is 50%
  # that's probably good enough for now :P
  project_hash=${project_hash:0:10}

  set-preferred-region

  if [[ "${ENABLE_DOCKER_REGISTRY_CACHE:-}" == "true" ]]; then
    DOCKER_REGISTRY_MIRROR_URL="https://mirror.gcr.io"
  fi

  SERVER_BINARY_TAR_HASH=$(sha1sum-file "${SERVER_BINARY_TAR}")
  if [[ -n "${KUBE_MANIFESTS_TAR:-}" ]]; then
    KUBE_MANIFESTS_TAR_HASH=$(sha1sum-file "${KUBE_MANIFESTS_TAR}")
  fi

  local server_binary_tar_urls=()
  local kube_manifest_tar_urls=()

  for region in "${PREFERRED_REGION[@]}"; do
    suffix="-${region}"
    if [[ "${suffix}" == "-us" ]]; then
      suffix=""
    fi
    local staging_bucket="gs://kubernetes-staging-${project_hash}${suffix}"

    # Ensure the buckets are created
    if ! gsutil ls "${staging_bucket}" >/dev/null; then
      echo "Creating ${staging_bucket}"
      gsutil mb -l "${region}" "${staging_bucket}"
    fi

    local staging_path="${staging_bucket}/${INSTANCE_PREFIX}-devel"

    echo "+++ Staging server tars to Google Storage: ${staging_path}"
    local server_binary_gs_url="${staging_path}/${SERVER_BINARY_TAR##*/}"
    copy-to-staging "${staging_path}" "${server_binary_gs_url}" "${SERVER_BINARY_TAR}" "${SERVER_BINARY_TAR_HASH}"

    # Convert from gs:// URL to an https:// URL
    server_binary_tar_urls+=("${server_binary_gs_url/gs:\/\//https://storage.googleapis.com/}")
    if [[ -n "${KUBE_MANIFESTS_TAR:-}" ]]; then
      local kube_manifests_gs_url="${staging_path}/${KUBE_MANIFESTS_TAR##*/}"
      copy-to-staging "${staging_path}" "${kube_manifests_gs_url}" "${KUBE_MANIFESTS_TAR}" "${KUBE_MANIFESTS_TAR_HASH}"
      # Convert from gs:// URL to an https:// URL
      kube_manifests_tar_urls+=("${kube_manifests_gs_url/gs:\/\//https://storage.googleapis.com/}")
    fi
  done

  SERVER_BINARY_TAR_URL=$(join_csv "${server_binary_tar_urls[@]}")
  if [[ -n "${KUBE_MANIFESTS_TAR:-}" ]]; then
    KUBE_MANIFESTS_TAR_URL=$(join_csv "${kube_manifests_tar_urls[@]}")
  fi
}

# Detect minions created in the minion group
#
# Assumed vars:
#   NODE_INSTANCE_PREFIX
# Vars set:
#   NODE_NAMES
#   INSTANCE_GROUPS
function detect-node-names() {
  detect-project
  INSTANCE_GROUPS=()
  INSTANCE_GROUPS+=($(gcloud compute instance-groups managed list \
    --project "${PROJECT}" \
    --filter "name ~ '${NODE_INSTANCE_PREFIX}-.+' AND zone:(${ZONE})" \
    --format='value(name)' || true))
  NODE_NAMES=()
  if [[ -n "${INSTANCE_GROUPS[@]:-}" ]]; then
    for group in "${INSTANCE_GROUPS[@]}"; do
      NODE_NAMES+=($(gcloud compute instance-groups managed list-instances \
        "${group}" --zone "${ZONE}" --project "${PROJECT}" \
        --format='value(instance)'))
    done
  fi
  # Add heapster node name to the list too (if it exists).
  if [[ -n "${HEAPSTER_MACHINE_TYPE:-}" ]]; then
    NODE_NAMES+=("${NODE_INSTANCE_PREFIX}-heapster")
  fi

  echo "INSTANCE_GROUPS=${INSTANCE_GROUPS[*]:-}" >&2
  echo "NODE_NAMES=${NODE_NAMES[*]:-}" >&2
}

# Detect the information about the minions
#
# Assumed vars:
#   ZONE
# Vars set:
#   NODE_NAMES
#   KUBE_NODE_IP_ADDRESSES (array)
function detect-nodes() {
  detect-project
  detect-node-names
  KUBE_NODE_IP_ADDRESSES=()
  for (( i=0; i<${#NODE_NAMES[@]}; i++)); do
    local node_ip=$(gcloud compute instances describe --project "${PROJECT}" --zone "${ZONE}" \
      "${NODE_NAMES[$i]}" --format='value(networkInterfaces[0].accessConfigs[0].natIP)')
    if [[ -z "${node_ip-}" ]] ; then
      echo "Did not find ${NODE_NAMES[$i]}" >&2
    else
      echo "Found ${NODE_NAMES[$i]} at ${node_ip}"
      KUBE_NODE_IP_ADDRESSES+=("${node_ip}")
    fi
  done
  if [[ -z "${KUBE_NODE_IP_ADDRESSES-}" ]]; then
    echo "Could not detect Kubernetes minion nodes.  Make sure you've launched a cluster with 'kube-up.sh'" >&2
    exit 1
  fi
}

# Detect the IP for the master
#
# Assumed vars:
#   MASTER_NAME
#   ZONE
#   REGION
# Vars set:
#   KUBE_MASTER
#   KUBE_MASTER_IP
function detect-master() {
  detect-project
  KUBE_MASTER=${MASTER_NAME}
  echo "Trying to find master named '${MASTER_NAME}'" >&2
  if [[ -z "${KUBE_MASTER_IP-}" ]]; then
    local master_address_name="${MASTER_NAME}-ip"
    echo "Looking for address '${master_address_name}'" >&2
    if ! KUBE_MASTER_IP=$(gcloud compute addresses describe "${master_address_name}" \
      --project "${PROJECT}" --region "${REGION}" -q --format='value(address)') || \
      [[ -z "${KUBE_MASTER_IP-}" ]]; then
      echo "Could not detect Kubernetes master node.  Make sure you've launched a cluster with 'kube-up.sh'" >&2
      exit 1
    fi
  fi
  echo "Using master: $KUBE_MASTER (external IP: $KUBE_MASTER_IP)" >&2
}

function load-or-gen-kube-bearertoken() {
  if [[ ! -z "${KUBE_CONTEXT:-}" ]]; then
    get-kubeconfig-bearertoken
  fi
  if [[ -z "${KUBE_BEARER_TOKEN:-}" ]]; then
    gen-kube-bearertoken
  fi
}

# Figure out which binary use on the server and assure it is available.
# If KUBE_VERSION is specified use binaries specified by it, otherwise
# use local dev binaries.
#
# Assumed vars:
#   KUBE_VERSION
#   KUBE_RELEASE_VERSION_REGEX
#   KUBE_CI_VERSION_REGEX
# Vars set:
#   KUBE_TAR_HASH
#   SERVER_BINARY_TAR_URL
#   SERVER_BINARY_TAR_HASH
function tars_from_version() {
  local sha1sum=""
  if which sha1sum >/dev/null 2>&1; then
    sha1sum="sha1sum"
  else
    sha1sum="shasum -a1"
  fi

  if [[ -z "${KUBE_VERSION-}" ]]; then
    find-release-tars
    upload-server-tars
  elif [[ ${KUBE_VERSION} =~ ${KUBE_RELEASE_VERSION_REGEX} ]]; then
    SERVER_BINARY_TAR_URL="https://storage.googleapis.com/kubernetes-release/release/${KUBE_VERSION}/kubernetes-server-linux-amd64.tar.gz"
    # TODO: Clean this up.
    KUBE_MANIFESTS_TAR_URL="${SERVER_BINARY_TAR_URL/server-linux-amd64/manifests}"
    KUBE_MANIFESTS_TAR_HASH=$(curl ${KUBE_MANIFESTS_TAR_URL} --silent --show-error | ${sha1sum} | awk '{print $1}')
  elif [[ ${KUBE_VERSION} =~ ${KUBE_CI_VERSION_REGEX} ]]; then
    SERVER_BINARY_TAR_URL="https://storage.googleapis.com/kubernetes-release-dev/ci/${KUBE_VERSION}/kubernetes-server-linux-amd64.tar.gz"
    # TODO: Clean this up.
    KUBE_MANIFESTS_TAR_URL="${SERVER_BINARY_TAR_URL/server-linux-amd64/manifests}"
    KUBE_MANIFESTS_TAR_HASH=$(curl ${KUBE_MANIFESTS_TAR_URL} --silent --show-error | ${sha1sum} | awk '{print $1}')
  else
    echo "Version doesn't match regexp" >&2
    exit 1
  fi
  if ! SERVER_BINARY_TAR_HASH=$(curl -Ss --fail "${SERVER_BINARY_TAR_URL}.sha1"); then
    echo "Failure trying to curl release .sha1"
  fi

  if ! curl -Ss --head "${SERVER_BINARY_TAR_URL}" >&/dev/null; then
    echo "Can't find release at ${SERVER_BINARY_TAR_URL}" >&2
    exit 1
  fi
}

# Reads kube-env metadata from master
#
# Assumed vars:
#   KUBE_MASTER
#   PROJECT
#   ZONE
function get-master-env() {
  # TODO(zmerlynn): Make this more reliable with retries.
  gcloud compute --project ${PROJECT} ssh --zone ${ZONE} ${KUBE_MASTER} --command \
    "curl --fail --silent -H 'Metadata-Flavor: Google' \
      'http://metadata/computeMetadata/v1/instance/attributes/kube-env'" 2>/dev/null
  gcloud compute --project ${PROJECT} ssh --zone ${ZONE} ${KUBE_MASTER} --command \
    "curl --fail --silent -H 'Metadata-Flavor: Google' \
      'http://metadata/computeMetadata/v1/instance/attributes/kube-master-certs'" 2>/dev/null
}

# Quote something appropriate for a yaml string.
#
# TODO(zmerlynn): Note that this function doesn't so much "quote" as
# "strip out quotes", and we really should be using a YAML library for
# this, but PyYAML isn't shipped by default, and *rant rant rant ... SIGH*
function yaml-quote {
  echo "'$(echo "${@:-}" | sed -e "s/'/''/g")'"
}

# Writes the cluster location into a temporary file.
# Assumed vars
#   ZONE
function write-cluster-location {
  cat >"${KUBE_TEMP}/cluster-location.txt" << EOF
${ZONE}
EOF
}

# Writes the cluster name into a temporary file.
# Assumed vars
#   CLUSTER_NAME
function write-cluster-name {
  cat >"${KUBE_TEMP}/cluster-name.txt" << EOF
${CLUSTER_NAME}
EOF
}

function write-master-env {
  # If the user requested that the master be part of the cluster, set the
  # environment variable to program the master kubelet to register itself.
  if [[ "${REGISTER_MASTER_KUBELET:-}" == "true" && -z "${KUBELET_APISERVER:-}" ]]; then
    KUBELET_APISERVER="${MASTER_NAME}"
  fi
  if [[ -z "${KUBERNETES_MASTER_NAME:-}" ]]; then
    KUBERNETES_MASTER_NAME="${MASTER_NAME}"
  fi

  construct-kubelet-flags true
  build-kube-env true "${KUBE_TEMP}/master-kube-env.yaml"
  build-kube-master-certs "${KUBE_TEMP}/kube-master-certs.yaml"
}

function write-node-env {
  if [[ -z "${KUBERNETES_MASTER_NAME:-}" ]]; then
    KUBERNETES_MASTER_NAME="${MASTER_NAME}"
  fi

  construct-kubelet-flags false
  build-kube-env false "${KUBE_TEMP}/node-kube-env.yaml"
}

function build-node-labels {
  local master=$1
  local node_labels=""
  if [[ "${KUBE_PROXY_DAEMONSET:-}" == "true" && "${master}" != "true" ]]; then
    # Add kube-proxy daemonset label to node to avoid situation during cluster
    # upgrade/downgrade when there are two instances of kube-proxy running on a node.
    node_labels="beta.kubernetes.io/kube-proxy-ds-ready=true"
  fi
  if [[ -n "${NODE_LABELS:-}" ]]; then
    node_labels="${node_labels:+${node_labels},}${NODE_LABELS}"
  fi
  if [[ -n "${NON_MASTER_NODE_LABELS:-}" && "${master}" != "true" ]]; then
    node_labels="${node_labels:+${node_labels},}${NON_MASTER_NODE_LABELS}"
  fi
  echo $node_labels
}

# $1: if 'true', we're rendering flags for a master, else a node
function construct-kubelet-flags {
  local master=$1
  local flags="${KUBELET_TEST_LOG_LEVEL:-"--v=2"} ${KUBELET_TEST_ARGS:-}"
  flags+=" --allow-privileged=true"
  flags+=" --cgroup-root=/"
  flags+=" --cloud-provider=gce"
  flags+=" --cluster-dns=${DNS_SERVER_IP}"
  flags+=" --cluster-domain=${DNS_DOMAIN}"
  flags+=" --pod-manifest-path=/etc/kubernetes/manifests"
  # Keep in sync with CONTAINERIZED_MOUNTER_HOME in configure-helper.sh
  flags+=" --experimental-mounter-path=/home/kubernetes/containerized_mounter/mounter"
  flags+=" --experimental-check-node-capabilities-before-mount=true"
  # Keep in sync with the mkdir command in configure-helper.sh (until the TODO is resolved)
  flags+=" --cert-dir=/var/lib/kubelet/pki/"

  if [[ "${master}" == "true" ]]; then
    flags+=" ${MASTER_KUBELET_TEST_ARGS:-}"
    flags+=" --enable-debugging-handlers=false"
    flags+=" --hairpin-mode=none"
    if [[ "${REGISTER_MASTER_KUBELET:-false}" == "true" ]]; then
      #TODO(mikedanese): allow static pods to start before creating a client
      #flags+=" --bootstrap-kubeconfig=/var/lib/kubelet/bootstrap-kubeconfig"
      #flags+=" --kubeconfig=/var/lib/kubelet/kubeconfig"
      flags+=" --kubeconfig=/var/lib/kubelet/bootstrap-kubeconfig"
      flags+=" --register-schedulable=false"
    else
      # Note: Standalone mode is used by GKE
      flags+=" --pod-cidr=${MASTER_IP_RANGE}"
    fi
  else # For nodes
    flags+=" ${NODE_KUBELET_TEST_ARGS:-}"
    flags+=" --enable-debugging-handlers=true"
    flags+=" --bootstrap-kubeconfig=/var/lib/kubelet/bootstrap-kubeconfig"
    flags+=" --kubeconfig=/var/lib/kubelet/kubeconfig"
    if [[ "${HAIRPIN_MODE:-}" == "promiscuous-bridge" ]] || \
       [[ "${HAIRPIN_MODE:-}" == "hairpin-veth" ]] || \
       [[ "${HAIRPIN_MODE:-}" == "none" ]]; then
      flags+=" --hairpin-mode=${HAIRPIN_MODE}"
    fi
    # Keep client-ca-file in sync with CA_CERT_BUNDLE_PATH in configure-helper.sh
    flags+=" --anonymous-auth=false --authorization-mode=Webhook --client-ca-file=/etc/srv/kubernetes/pki/ca-certificates.crt"
  fi
  # Network plugin
  if [[ -n "${NETWORK_PROVIDER:-}" || -n "${NETWORK_POLICY_PROVIDER:-}" ]]; then
    flags+=" --cni-bin-dir=/home/kubernetes/bin"
    if [[ "${NETWORK_POLICY_PROVIDER:-}" == "calico" ]]; then
      # Calico uses CNI always.
      # Note that network policy won't work for master node.
      if [[ "${master}" == "true" ]]; then
        flags+=" --network-plugin=${NETWORK_PROVIDER}"
      else
        flags+=" --network-plugin=cni"
      fi
    else
      # Otherwise use the configured value.
      flags+=" --network-plugin=${NETWORK_PROVIDER}"
    fi
  fi
  if [[ -n "${NON_MASQUERADE_CIDR:-}" ]]; then
    flags+=" --non-masquerade-cidr=${NON_MASQUERADE_CIDR}"
  fi
  flags+=" --volume-plugin-dir=${VOLUME_PLUGIN_DIR}"
  # Note: ENABLE_MANIFEST_URL is used by GKE
  if [[ "${ENABLE_MANIFEST_URL:-}" == "true" ]]; then
    flags+=" --manifest-url=${MANIFEST_URL}"
    flags+=" --manifest-url-header=${MANIFEST_URL_HEADER}"
  fi
  if [[ -n "${ENABLE_CUSTOM_METRICS:-}" ]]; then
    flags+=" --enable-custom-metrics=${ENABLE_CUSTOM_METRICS}"
  fi
  local node_labels=$(build-node-labels ${master})
  if [[ -n "${node_labels:-}" ]]; then
    flags+=" --node-labels=${node_labels}"
  fi
  if [[ -n "${NODE_TAINTS:-}" ]]; then
    flags+=" --register-with-taints=${NODE_TAINTS}"
  fi
  if [[ -n "${EVICTION_HARD:-}" ]]; then
    flags+=" --eviction-hard=${EVICTION_HARD}"
  fi
  if [[ -n "${FEATURE_GATES:-}" ]]; then
    flags+=" --feature-gates=${FEATURE_GATES}"
  fi
  # TODO(mtaufen): ROTATE_CERTIFICATES seems unused; delete it?
  if [[ -n "${ROTATE_CERTIFICATES:-}" ]]; then
    flags+=" --rotate-certificates=true"
  fi
  if [[ -n "${CONTAINER_RUNTIME:-}" ]]; then
    flags+=" --container-runtime=${CONTAINER_RUNTIME}"
  fi
  # TODO(mtaufen): CONTAINER_RUNTIME_ENDPOINT seems unused; delete it?
  if [[ -n "${CONTAINER_RUNTIME_ENDPOINT:-}" ]]; then
    flags+=" --container-runtime-endpoint=${CONTAINER_RUNTIME_ENDPOINT}"
  fi

  KUBELET_ARGS="${flags}"
}

function build-kube-master-certs {
  local file=$1
  rm -f ${file}
  cat >$file <<EOF
KUBEAPISERVER_CERT: $(yaml-quote ${KUBEAPISERVER_CERT_BASE64:-})
KUBEAPISERVER_KEY: $(yaml-quote ${KUBEAPISERVER_KEY_BASE64:-})
CA_KEY: $(yaml-quote ${CA_KEY_BASE64:-})
AGGREGATOR_CA_KEY: $(yaml-quote ${AGGREGATOR_CA_KEY_BASE64:-})
REQUESTHEADER_CA_CERT: $(yaml-quote ${REQUESTHEADER_CA_CERT_BASE64:-})
PROXY_CLIENT_CERT: $(yaml-quote ${PROXY_CLIENT_CERT_BASE64:-})
PROXY_CLIENT_KEY: $(yaml-quote ${PROXY_CLIENT_KEY_BASE64:-})
EOF
}

# $1: if 'true', we're building a master yaml, else a node
function build-kube-env {
  local master=$1
  local file=$2

  local server_binary_tar_url=$SERVER_BINARY_TAR_URL
  local kube_manifests_tar_url="${KUBE_MANIFESTS_TAR_URL:-}"
  if [[ "${master}" == "true" && "${MASTER_OS_DISTRIBUTION}" == "ubuntu" ]] || \
     [[ "${master}" == "false" && ("${NODE_OS_DISTRIBUTION}" == "ubuntu" || "${NODE_OS_DISTRIBUTION}" == "custom") ]]; then
    # TODO: Support fallback .tar.gz settings on Container Linux
    server_binary_tar_url=$(split_csv "${SERVER_BINARY_TAR_URL}")
    kube_manifests_tar_url=$(split_csv "${KUBE_MANIFESTS_TAR_URL}")
  fi

  rm -f ${file}
  cat >$file <<EOF
CLUSTER_NAME: $(yaml-quote ${CLUSTER_NAME})
ENV_TIMESTAMP: $(yaml-quote $(date -u +%Y-%m-%dT%T%z))
INSTANCE_PREFIX: $(yaml-quote ${INSTANCE_PREFIX})
NODE_INSTANCE_PREFIX: $(yaml-quote ${NODE_INSTANCE_PREFIX})
NODE_TAGS: $(yaml-quote ${NODE_TAGS:-})
NODE_NETWORK: $(yaml-quote ${NETWORK:-})
NODE_SUBNETWORK: $(yaml-quote ${SUBNETWORK:-})
CLUSTER_IP_RANGE: $(yaml-quote ${CLUSTER_IP_RANGE:-10.244.0.0/16})
SERVER_BINARY_TAR_URL: $(yaml-quote ${server_binary_tar_url})
SERVER_BINARY_TAR_HASH: $(yaml-quote ${SERVER_BINARY_TAR_HASH})
PROJECT_ID: $(yaml-quote ${PROJECT})
NETWORK_PROJECT_ID: $(yaml-quote ${NETWORK_PROJECT})
SERVICE_CLUSTER_IP_RANGE: $(yaml-quote ${SERVICE_CLUSTER_IP_RANGE})
KUBERNETES_MASTER_NAME: $(yaml-quote ${KUBERNETES_MASTER_NAME})
ALLOCATE_NODE_CIDRS: $(yaml-quote ${ALLOCATE_NODE_CIDRS:-false})
ENABLE_CLUSTER_MONITORING: $(yaml-quote ${ENABLE_CLUSTER_MONITORING:-none})
ENABLE_METRICS_SERVER: $(yaml-quote ${ENABLE_METRICS_SERVER:-false})
ENABLE_METADATA_AGENT: $(yaml-quote ${ENABLE_METADATA_AGENT:-none})
METADATA_AGENT_CPU_REQUEST: $(yaml-quote ${METADATA_AGENT_CPU_REQUEST:-})
METADATA_AGENT_MEMORY_REQUEST: $(yaml-quote ${METADATA_AGENT_MEMORY_REQUEST:-})
METADATA_AGENT_CLUSTER_LEVEL_CPU_REQUEST: $(yaml-quote ${METADATA_AGENT_CLUSTER_LEVEL_CPU_REQUEST:-})
METADATA_AGENT_CLUSTER_LEVEL_MEMORY_REQUEST: $(yaml-quote ${METADATA_AGENT_CLUSTER_LEVEL_MEMORY_REQUEST:-})
DOCKER_REGISTRY_MIRROR_URL: $(yaml-quote ${DOCKER_REGISTRY_MIRROR_URL:-})
ENABLE_L7_LOADBALANCING: $(yaml-quote ${ENABLE_L7_LOADBALANCING:-none})
ENABLE_CLUSTER_LOGGING: $(yaml-quote ${ENABLE_CLUSTER_LOGGING:-false})
ENABLE_CLUSTER_UI: $(yaml-quote ${ENABLE_CLUSTER_UI:-false})
ENABLE_NODE_PROBLEM_DETECTOR: $(yaml-quote ${ENABLE_NODE_PROBLEM_DETECTOR:-none})
NODE_PROBLEM_DETECTOR_VERSION: $(yaml-quote ${NODE_PROBLEM_DETECTOR_VERSION:-})
NODE_PROBLEM_DETECTOR_TAR_HASH: $(yaml-quote ${NODE_PROBLEM_DETECTOR_TAR_HASH:-})
ENABLE_NODE_LOGGING: $(yaml-quote ${ENABLE_NODE_LOGGING:-false})
ENABLE_RESCHEDULER: $(yaml-quote ${ENABLE_RESCHEDULER:-false})
LOGGING_DESTINATION: $(yaml-quote ${LOGGING_DESTINATION:-})
ELASTICSEARCH_LOGGING_REPLICAS: $(yaml-quote ${ELASTICSEARCH_LOGGING_REPLICAS:-})
ENABLE_CLUSTER_DNS: $(yaml-quote ${ENABLE_CLUSTER_DNS:-false})
CLUSTER_DNS_CORE_DNS: $(yaml-quote ${CLUSTER_DNS_CORE_DNS:-false})
DNS_SERVER_IP: $(yaml-quote ${DNS_SERVER_IP:-})
DNS_DOMAIN: $(yaml-quote ${DNS_DOMAIN:-})
ENABLE_DNS_HORIZONTAL_AUTOSCALER: $(yaml-quote ${ENABLE_DNS_HORIZONTAL_AUTOSCALER:-false})
KUBE_PROXY_DAEMONSET: $(yaml-quote ${KUBE_PROXY_DAEMONSET:-false})
KUBE_PROXY_TOKEN: $(yaml-quote ${KUBE_PROXY_TOKEN:-})
KUBE_PROXY_MODE: $(yaml-quote ${KUBE_PROXY_MODE:-iptables})
NODE_PROBLEM_DETECTOR_TOKEN: $(yaml-quote ${NODE_PROBLEM_DETECTOR_TOKEN:-})
ADMISSION_CONTROL: $(yaml-quote ${ADMISSION_CONTROL:-})
ENABLE_POD_SECURITY_POLICY: $(yaml-quote ${ENABLE_POD_SECURITY_POLICY:-})
MASTER_IP_RANGE: $(yaml-quote ${MASTER_IP_RANGE})
RUNTIME_CONFIG: $(yaml-quote ${RUNTIME_CONFIG})
CA_CERT: $(yaml-quote ${CA_CERT_BASE64:-})
KUBELET_CERT: $(yaml-quote ${KUBELET_CERT_BASE64:-})
KUBELET_KEY: $(yaml-quote ${KUBELET_KEY_BASE64:-})
NETWORK_PROVIDER: $(yaml-quote ${NETWORK_PROVIDER:-})
NETWORK_POLICY_PROVIDER: $(yaml-quote ${NETWORK_POLICY_PROVIDER:-})
PREPULL_E2E_IMAGES: $(yaml-quote ${PREPULL_E2E_IMAGES:-})
HAIRPIN_MODE: $(yaml-quote ${HAIRPIN_MODE:-})
E2E_STORAGE_TEST_ENVIRONMENT: $(yaml-quote ${E2E_STORAGE_TEST_ENVIRONMENT:-})
KUBE_DOCKER_REGISTRY: $(yaml-quote ${KUBE_DOCKER_REGISTRY:-})
KUBE_ADDON_REGISTRY: $(yaml-quote ${KUBE_ADDON_REGISTRY:-})
MULTIZONE: $(yaml-quote ${MULTIZONE:-})
NON_MASQUERADE_CIDR: $(yaml-quote ${NON_MASQUERADE_CIDR:-})
ENABLE_DEFAULT_STORAGE_CLASS: $(yaml-quote ${ENABLE_DEFAULT_STORAGE_CLASS:-})
ENABLE_APISERVER_BASIC_AUDIT: $(yaml-quote ${ENABLE_APISERVER_BASIC_AUDIT:-})
ENABLE_APISERVER_ADVANCED_AUDIT: $(yaml-quote ${ENABLE_APISERVER_ADVANCED_AUDIT:-})
ENABLE_CACHE_MUTATION_DETECTOR: $(yaml-quote ${ENABLE_CACHE_MUTATION_DETECTOR:-false})
ENABLE_PATCH_CONVERSION_DETECTOR: $(yaml-quote ${ENABLE_PATCH_CONVERSION_DETECTOR:-false})
ADVANCED_AUDIT_POLICY: $(yaml-quote ${ADVANCED_AUDIT_POLICY:-})
ADVANCED_AUDIT_BACKEND: $(yaml-quote ${ADVANCED_AUDIT_BACKEND:-log})
ADVANCED_AUDIT_LOG_MODE: $(yaml-quote ${ADVANCED_AUDIT_LOG_MODE:-})
ADVANCED_AUDIT_LOG_BUFFER_SIZE: $(yaml-quote ${ADVANCED_AUDIT_LOG_BUFFER_SIZE:-})
ADVANCED_AUDIT_LOG_MAX_BATCH_SIZE: $(yaml-quote ${ADVANCED_AUDIT_LOG_MAX_BATCH_SIZE:-})
ADVANCED_AUDIT_LOG_MAX_BATCH_WAIT: $(yaml-quote ${ADVANCED_AUDIT_LOG_MAX_BATCH_WAIT:-})
ADVANCED_AUDIT_LOG_THROTTLE_QPS: $(yaml-quote ${ADVANCED_AUDIT_LOG_THROTTLE_QPS:-})
ADVANCED_AUDIT_LOG_THROTTLE_BURST: $(yaml-quote ${ADVANCED_AUDIT_LOG_THROTTLE_BURST:-})
ADVANCED_AUDIT_LOG_INITIAL_BACKOFF: $(yaml-quote ${ADVANCED_AUDIT_LOG_INITIAL_BACKOFF:-})
ADVANCED_AUDIT_WEBHOOK_MODE: $(yaml-quote ${ADVANCED_AUDIT_WEBHOOK_MODE:-})
ADVANCED_AUDIT_WEBHOOK_BUFFER_SIZE: $(yaml-quote ${ADVANCED_AUDIT_WEBHOOK_BUFFER_SIZE:-})
ADVANCED_AUDIT_WEBHOOK_MAX_BATCH_SIZE: $(yaml-quote ${ADVANCED_AUDIT_WEBHOOK_MAX_BATCH_SIZE:-})
ADVANCED_AUDIT_WEBHOOK_MAX_BATCH_WAIT: $(yaml-quote ${ADVANCED_AUDIT_WEBHOOK_MAX_BATCH_WAIT:-})
ADVANCED_AUDIT_WEBHOOK_THROTTLE_QPS: $(yaml-quote ${ADVANCED_AUDIT_WEBHOOK_THROTTLE_QPS:-})
ADVANCED_AUDIT_WEBHOOK_THROTTLE_BURST: $(yaml-quote ${ADVANCED_AUDIT_WEBHOOK_THROTTLE_BURST:-})
ADVANCED_AUDIT_WEBHOOK_INITIAL_BACKOFF: $(yaml-quote ${ADVANCED_AUDIT_WEBHOOK_INITIAL_BACKOFF:-})
GCE_API_ENDPOINT: $(yaml-quote ${GCE_API_ENDPOINT:-})
GCE_GLBC_IMAGE: $(yaml-quote ${GCE_GLBC_IMAGE:-})
PROMETHEUS_TO_SD_ENDPOINT: $(yaml-quote ${PROMETHEUS_TO_SD_ENDPOINT:-})
PROMETHEUS_TO_SD_PREFIX: $(yaml-quote ${PROMETHEUS_TO_SD_PREFIX:-})
ENABLE_PROMETHEUS_TO_SD: $(yaml-quote ${ENABLE_PROMETHEUS_TO_SD:-false})
ENABLE_POD_PRIORITY: $(yaml-quote ${ENABLE_POD_PRIORITY:-})
CONTAINER_RUNTIME: $(yaml-quote ${CONTAINER_RUNTIME:-})
CONTAINER_RUNTIME_ENDPOINT: $(yaml-quote ${CONTAINER_RUNTIME_ENDPOINT:-})
CONTAINER_RUNTIME_NAME: $(yaml-quote ${CONTAINER_RUNTIME_NAME:-})
NODE_LOCAL_SSDS_EXT: $(yaml-quote ${NODE_LOCAL_SSDS_EXT:-})
LOAD_IMAGE_COMMAND: $(yaml-quote ${LOAD_IMAGE_COMMAND:-})
ZONE: $(yaml-quote ${ZONE})
VOLUME_PLUGIN_DIR: $(yaml-quote ${VOLUME_PLUGIN_DIR})
KUBELET_ARGS: $(yaml-quote ${KUBELET_ARGS})
EOF
  if [[ "${master}" == "true" && "${MASTER_OS_DISTRIBUTION}" == "gci" ]] || \
     [[ "${master}" == "false" && "${NODE_OS_DISTRIBUTION}" == "gci" ]]  || \
     [[ "${master}" == "true" && "${MASTER_OS_DISTRIBUTION}" == "cos" ]] || \
     [[ "${master}" == "false" && "${NODE_OS_DISTRIBUTION}" == "cos" ]]; then
    cat >>$file <<EOF
REMOUNT_VOLUME_PLUGIN_DIR: $(yaml-quote ${REMOUNT_VOLUME_PLUGIN_DIR:-true})
EOF
  fi
  if [ -n "${KUBE_APISERVER_REQUEST_TIMEOUT:-}" ]; then
    cat >>$file <<EOF
KUBE_APISERVER_REQUEST_TIMEOUT: $(yaml-quote ${KUBE_APISERVER_REQUEST_TIMEOUT})
EOF
  fi
  if [ -n "${TERMINATED_POD_GC_THRESHOLD:-}" ]; then
    cat >>$file <<EOF
TERMINATED_POD_GC_THRESHOLD: $(yaml-quote ${TERMINATED_POD_GC_THRESHOLD})
EOF
  fi
  if [[ "${master}" == "true" && ("${MASTER_OS_DISTRIBUTION}" == "trusty" || "${MASTER_OS_DISTRIBUTION}" == "gci" || "${MASTER_OS_DISTRIBUTION}" == "ubuntu") ]] || \
     [[ "${master}" == "false" && ("${NODE_OS_DISTRIBUTION}" == "trusty" || "${NODE_OS_DISTRIBUTION}" == "gci" || "${NODE_OS_DISTRIBUTION}" = "ubuntu" || "${NODE_OS_DISTRIBUTION}" = "custom") ]] ; then
    cat >>$file <<EOF
KUBE_MANIFESTS_TAR_URL: $(yaml-quote ${kube_manifests_tar_url})
KUBE_MANIFESTS_TAR_HASH: $(yaml-quote ${KUBE_MANIFESTS_TAR_HASH})
EOF
  fi
  if [ -n "${TEST_CLUSTER:-}" ]; then
    cat >>$file <<EOF
TEST_CLUSTER: $(yaml-quote ${TEST_CLUSTER})
EOF
  fi
  if [ -n "${DOCKER_TEST_LOG_LEVEL:-}" ]; then
      cat >>$file <<EOF
DOCKER_TEST_LOG_LEVEL: $(yaml-quote ${DOCKER_TEST_LOG_LEVEL})
EOF
  fi
  if [ -n "${DOCKER_LOG_DRIVER:-}" ]; then
      cat >>$file <<EOF
DOCKER_LOG_DRIVER: $(yaml-quote ${DOCKER_LOG_DRIVER})
EOF
  fi
  if [ -n "${DOCKER_LOG_MAX_SIZE:-}" ]; then
      cat >>$file <<EOF
DOCKER_LOG_MAX_SIZE: $(yaml-quote ${DOCKER_LOG_MAX_SIZE})
EOF
  fi
  if [ -n "${DOCKER_LOG_MAX_FILE:-}" ]; then
      cat >>$file <<EOF
DOCKER_LOG_MAX_FILE: $(yaml-quote ${DOCKER_LOG_MAX_FILE})
EOF
  fi
  if [ -n "${FEATURE_GATES:-}" ]; then
    cat >>$file <<EOF
FEATURE_GATES: $(yaml-quote ${FEATURE_GATES})
EOF
  fi
  if [ -n "${PROVIDER_VARS:-}" ]; then
    local var_name
    local var_value

    for var_name in ${PROVIDER_VARS}; do
      eval "local var_value=\$(yaml-quote \${${var_name}})"
      cat >>$file <<EOF
${var_name}: ${var_value}
EOF
    done
  fi

  if [[ "${master}" == "true" ]]; then
    # Master-only env vars.
    cat >>$file <<EOF
KUBERNETES_MASTER: $(yaml-quote "true")
KUBE_USER: $(yaml-quote ${KUBE_USER})
KUBE_PASSWORD: $(yaml-quote ${KUBE_PASSWORD})
KUBE_BEARER_TOKEN: $(yaml-quote ${KUBE_BEARER_TOKEN})
MASTER_CERT: $(yaml-quote ${MASTER_CERT_BASE64:-})
MASTER_KEY: $(yaml-quote ${MASTER_KEY_BASE64:-})
KUBECFG_CERT: $(yaml-quote ${KUBECFG_CERT_BASE64:-})
KUBECFG_KEY: $(yaml-quote ${KUBECFG_KEY_BASE64:-})
KUBELET_APISERVER: $(yaml-quote ${KUBELET_APISERVER:-})
NUM_NODES: $(yaml-quote ${NUM_NODES})
STORAGE_BACKEND: $(yaml-quote ${STORAGE_BACKEND:-etcd3})
STORAGE_MEDIA_TYPE: $(yaml-quote ${STORAGE_MEDIA_TYPE:-})
ENABLE_GARBAGE_COLLECTOR: $(yaml-quote ${ENABLE_GARBAGE_COLLECTOR:-})
ENABLE_LEGACY_ABAC: $(yaml-quote ${ENABLE_LEGACY_ABAC:-})
MASTER_ADVERTISE_ADDRESS: $(yaml-quote ${MASTER_ADVERTISE_ADDRESS:-})
ETCD_CA_KEY: $(yaml-quote ${ETCD_CA_KEY_BASE64:-})
ETCD_CA_CERT: $(yaml-quote ${ETCD_CA_CERT_BASE64:-})
ETCD_PEER_KEY: $(yaml-quote ${ETCD_PEER_KEY_BASE64:-})
ETCD_PEER_CERT: $(yaml-quote ${ETCD_PEER_CERT_BASE64:-})
EOF
    if [[ "${ENABLE_TOKENREQUEST:-}" == "true" ]]; then
      cat >>$file <<EOF
SERVICEACCOUNT_ISSUER: $(yaml-quote ${SERVICEACCOUNT_ISSUER:-})
SERVICEACCOUNT_API_AUDIENCES: $(yaml-quote ${SERVICEACCOUNT_API_AUDIENCES:-})
EOF
    fi
    # KUBE_APISERVER_REQUEST_TIMEOUT_SEC (if set) controls the --request-timeout
    # flag
    if [ -n "${KUBE_APISERVER_REQUEST_TIMEOUT_SEC:-}" ]; then
      cat >>$file <<EOF
KUBE_APISERVER_REQUEST_TIMEOUT_SEC: $(yaml-quote ${KUBE_APISERVER_REQUEST_TIMEOUT_SEC})
EOF
    fi
    # ETCD_IMAGE (if set) allows to use a custom etcd image.
    if [ -n "${ETCD_IMAGE:-}" ]; then
      cat >>$file <<EOF
ETCD_IMAGE: $(yaml-quote ${ETCD_IMAGE})
EOF
    fi
    # ETCD_DOCKER_REPOSITORY (if set) allows to use a custom etcd docker repository to pull the etcd image from.
    if [ -n "${ETCD_DOCKER_REPOSITORY:-}" ]; then
      cat >>$file <<EOF
ETCD_DOCKER_REPOSITORY: $(yaml-quote ${ETCD_DOCKER_REPOSITORY})
EOF
    fi
    # ETCD_VERSION (if set) allows you to use custom version of etcd.
    # The main purpose of using it may be rollback of etcd v3 API,
    # where we need 3.0.* image, but are rolling back to 2.3.7.
    if [ -n "${ETCD_VERSION:-}" ]; then
      cat >>$file <<EOF
ETCD_VERSION: $(yaml-quote ${ETCD_VERSION})
EOF
    fi
    if [ -n "${ETCD_HOSTNAME:-}" ]; then
      cat >>$file <<EOF
ETCD_HOSTNAME: $(yaml-quote ${ETCD_HOSTNAME})
EOF
    fi
    if [ -n "${ETCD_LIVENESS_PROBE_INITIAL_DELAY_SEC:-}" ]; then
      cat >>$file <<EOF
ETCD_LIVENESS_PROBE_INITIAL_DELAY_SEC: $(yaml-quote ${ETCD_LIVENESS_PROBE_INITIAL_DELAY_SEC})
EOF
    fi
    if [ -n "${KUBE_APISERVER_LIVENESS_PROBE_INITIAL_DELAY_SEC:-}" ]; then
      cat >>$file <<EOF
KUBE_APISERVER_LIVENESS_PROBE_INITIAL_DELAY_SEC: $(yaml-quote ${KUBE_APISERVER_LIVENESS_PROBE_INITIAL_DELAY_SEC})
EOF
    fi
    if [ -n "${ETCD_COMPACTION_INTERVAL_SEC:-}" ]; then
      cat >>$file <<EOF
ETCD_COMPACTION_INTERVAL_SEC: $(yaml-quote ${ETCD_COMPACTION_INTERVAL_SEC})
EOF
    fi
    if [ -n "${ETCD_QUOTA_BACKEND_BYTES:-}" ]; then
      cat >>$file <<EOF
ETCD_QUOTA_BACKEND_BYTES: $(yaml-quote ${ETCD_QUOTA_BACKEND_BYTES})
EOF
    fi
    if [ -n "${APISERVER_TEST_ARGS:-}" ]; then
      cat >>$file <<EOF
APISERVER_TEST_ARGS: $(yaml-quote ${APISERVER_TEST_ARGS})
EOF
    fi
    if [ -n "${CONTROLLER_MANAGER_TEST_ARGS:-}" ]; then
      cat >>$file <<EOF
CONTROLLER_MANAGER_TEST_ARGS: $(yaml-quote ${CONTROLLER_MANAGER_TEST_ARGS})
EOF
    fi
    if [ -n "${CONTROLLER_MANAGER_TEST_LOG_LEVEL:-}" ]; then
      cat >>$file <<EOF
CONTROLLER_MANAGER_TEST_LOG_LEVEL: $(yaml-quote ${CONTROLLER_MANAGER_TEST_LOG_LEVEL})
EOF
    fi
    if [ -n "${SCHEDULER_TEST_ARGS:-}" ]; then
      cat >>$file <<EOF
SCHEDULER_TEST_ARGS: $(yaml-quote ${SCHEDULER_TEST_ARGS})
EOF
    fi
    if [ -n "${SCHEDULER_TEST_LOG_LEVEL:-}" ]; then
      cat >>$file <<EOF
SCHEDULER_TEST_LOG_LEVEL: $(yaml-quote ${SCHEDULER_TEST_LOG_LEVEL})
EOF
    fi
    if [ -n "${INITIAL_ETCD_CLUSTER:-}" ]; then
      cat >>$file <<EOF
INITIAL_ETCD_CLUSTER: $(yaml-quote ${INITIAL_ETCD_CLUSTER})
EOF
    fi
    if [ -n "${INITIAL_ETCD_CLUSTER_STATE:-}" ]; then
      cat >>$file <<EOF
INITIAL_ETCD_CLUSTER_STATE: $(yaml-quote ${INITIAL_ETCD_CLUSTER_STATE})
EOF
    fi
    if [ -n "${ETCD_QUORUM_READ:-}" ]; then
      cat >>$file <<EOF
ETCD_QUORUM_READ: $(yaml-quote ${ETCD_QUORUM_READ})
EOF
    fi
    if [ -n "${CLUSTER_SIGNING_DURATION:-}" ]; then
      cat >>$file <<EOF
CLUSTER_SIGNING_DURATION: $(yaml-quote ${CLUSTER_SIGNING_DURATION})
EOF
    fi
    if [[ "${NODE_ACCELERATORS:-}" == *"type=nvidia"* ]]; then
      cat >>$file <<EOF
ENABLE_NVIDIA_GPU_DEVICE_PLUGIN: $(yaml-quote "true")
EOF
    fi
    if [ -n "${ADDON_MANAGER_LEADER_ELECTION:-}" ]; then
      cat >>$file <<EOF
ADDON_MANAGER_LEADER_ELECTION: $(yaml-quote ${ADDON_MANAGER_LEADER_ELECTION})
EOF
    fi

  else
    # Node-only env vars.
    cat >>$file <<EOF
KUBERNETES_MASTER: $(yaml-quote "false")
EXTRA_DOCKER_OPTS: $(yaml-quote ${EXTRA_DOCKER_OPTS:-})
EOF
    if [ -n "${KUBEPROXY_TEST_ARGS:-}" ]; then
      cat >>$file <<EOF
KUBEPROXY_TEST_ARGS: $(yaml-quote ${KUBEPROXY_TEST_ARGS})
EOF
    fi
    if [ -n "${KUBEPROXY_TEST_LOG_LEVEL:-}" ]; then
      cat >>$file <<EOF
KUBEPROXY_TEST_LOG_LEVEL: $(yaml-quote ${KUBEPROXY_TEST_LOG_LEVEL})
EOF
    fi
  fi
  if [[ "${ENABLE_CLUSTER_AUTOSCALER}" == "true" ]]; then
      cat >>$file <<EOF
ENABLE_CLUSTER_AUTOSCALER: $(yaml-quote ${ENABLE_CLUSTER_AUTOSCALER})
AUTOSCALER_MIG_CONFIG: $(yaml-quote ${AUTOSCALER_MIG_CONFIG})
AUTOSCALER_EXPANDER_CONFIG: $(yaml-quote ${AUTOSCALER_EXPANDER_CONFIG})
EOF
      if [[ "${master}" == "false" ]]; then
          # TODO(kubernetes/autoscaler#718): AUTOSCALER_ENV_VARS is a hotfix for cluster autoscaler,
          # which reads the kube-env to determine the shape of a node and was broken by #60020.
          # This should be removed as soon as a more reliable source of information is available!
          local node_labels=$(build-node-labels false)
          local node_taints="${NODE_TAINTS:-}"
          local autoscaler_env_vars="node_labels=${node_labels};node_taints=${node_taints}"
          cat >>$file <<EOF
AUTOSCALER_ENV_VARS: $(yaml-quote ${autoscaler_env_vars})
EOF
      fi
  fi
  if [ -n "${SCHEDULING_ALGORITHM_PROVIDER:-}" ]; then
    cat >>$file <<EOF
SCHEDULING_ALGORITHM_PROVIDER: $(yaml-quote ${SCHEDULING_ALGORITHM_PROVIDER})
EOF
  fi
}

function sha1sum-file() {
  if which sha1sum >/dev/null 2>&1; then
    sha1sum "$1" | awk '{ print $1 }'
  else
    shasum -a1 "$1" | awk '{ print $1 }'
  fi
}

# Create certificate pairs for the cluster.
# $1: The public IP for the master.
#
# These are used for static cert distribution (e.g. static clustering) at
# cluster creation time. This will be obsoleted once we implement dynamic
# clustering.
#
# The following certificate pairs are created:
#
#  - ca (the cluster's certificate authority)
#  - server
#  - kubelet
#  - kubecfg (for kubectl)
#
# TODO(roberthbailey): Replace easyrsa with a simple Go program to generate
# the certs that we need.
#
# Assumed vars
#   KUBE_TEMP
#   MASTER_NAME
#
# Vars set:
#   CERT_DIR
#   CA_CERT_BASE64
#   MASTER_CERT_BASE64
#   MASTER_KEY_BASE64
#   KUBELET_CERT_BASE64
#   KUBELET_KEY_BASE64
#   KUBECFG_CERT_BASE64
#   KUBECFG_KEY_BASE64
function create-certs {
  local -r primary_cn="${1}"

  # Determine extra certificate names for master
  local octets=($(echo "${SERVICE_CLUSTER_IP_RANGE}" | sed -e 's|/.*||' -e 's/\./ /g'))
  ((octets[3]+=1))
  local -r service_ip=$(echo "${octets[*]}" | sed 's/ /./g')
  local sans=""
  for extra in $@; do
    if [[ -n "${extra}" ]]; then
      sans="${sans}IP:${extra},"
    fi
  done
  sans="${sans}IP:${service_ip},DNS:kubernetes,DNS:kubernetes.default,DNS:kubernetes.default.svc,DNS:kubernetes.default.svc.${DNS_DOMAIN},DNS:${MASTER_NAME}"

  echo "Generating certs for alternate-names: ${sans}"

  setup-easyrsa
  PRIMARY_CN="${primary_cn}" SANS="${sans}" generate-certs
  AGGREGATOR_PRIMARY_CN="${primary_cn}" AGGREGATOR_SANS="${sans}" generate-aggregator-certs

  # By default, linux wraps base64 output every 76 cols, so we use 'tr -d' to remove whitespaces.
  # Note 'base64 -w0' doesn't work on Mac OS X, which has different flags.
  CA_KEY_BASE64=$(cat "${CERT_DIR}/pki/private/ca.key" | base64 | tr -d '\r\n')
  CA_CERT_BASE64=$(cat "${CERT_DIR}/pki/ca.crt" | base64 | tr -d '\r\n')
  MASTER_CERT_BASE64=$(cat "${CERT_DIR}/pki/issued/${MASTER_NAME}.crt" | base64 | tr -d '\r\n')
  MASTER_KEY_BASE64=$(cat "${CERT_DIR}/pki/private/${MASTER_NAME}.key" | base64 | tr -d '\r\n')
  KUBELET_CERT_BASE64=$(cat "${CERT_DIR}/pki/issued/kubelet.crt" | base64 | tr -d '\r\n')
  KUBELET_KEY_BASE64=$(cat "${CERT_DIR}/pki/private/kubelet.key" | base64 | tr -d '\r\n')
  KUBECFG_CERT_BASE64=$(cat "${CERT_DIR}/pki/issued/kubecfg.crt" | base64 | tr -d '\r\n')
  KUBECFG_KEY_BASE64=$(cat "${CERT_DIR}/pki/private/kubecfg.key" | base64 | tr -d '\r\n')
  KUBEAPISERVER_CERT_BASE64=$(cat "${CERT_DIR}/pki/issued/kube-apiserver.crt" | base64 | tr -d '\r\n')
  KUBEAPISERVER_KEY_BASE64=$(cat "${CERT_DIR}/pki/private/kube-apiserver.key" | base64 | tr -d '\r\n')

  # Setting up an addition directory (beyond pki) as it is the simplest way to
  # ensure we get a different CA pair to sign the proxy-client certs and which
  # we can send CA public key to the user-apiserver to validate communication.
  AGGREGATOR_CA_KEY_BASE64=$(cat "${AGGREGATOR_CERT_DIR}/pki/private/ca.key" | base64 | tr -d '\r\n')
  REQUESTHEADER_CA_CERT_BASE64=$(cat "${AGGREGATOR_CERT_DIR}/pki/ca.crt" | base64 | tr -d '\r\n')
  PROXY_CLIENT_CERT_BASE64=$(cat "${AGGREGATOR_CERT_DIR}/pki/issued/proxy-client.crt" | base64 | tr -d '\r\n')
  PROXY_CLIENT_KEY_BASE64=$(cat "${AGGREGATOR_CERT_DIR}/pki/private/proxy-client.key" | base64 | tr -d '\r\n')
}

# Set up easy-rsa directory structure.
#
# Assumed vars
#   KUBE_TEMP
#
# Vars set:
#   CERT_DIR
#   AGGREGATOR_CERT_DIR
function setup-easyrsa {
  local -r cert_create_debug_output=$(mktemp "${KUBE_TEMP}/cert_create_debug_output.XXX")
  # Note: This was heavily cribbed from make-ca-cert.sh
  (set -x
    cd "${KUBE_TEMP}"
    curl -L -O --connect-timeout 20 --retry 6 --retry-delay 2 https://storage.googleapis.com/kubernetes-release/easy-rsa/easy-rsa.tar.gz
    tar xzf easy-rsa.tar.gz
    mkdir easy-rsa-master/kubelet
    cp -r easy-rsa-master/easyrsa3/* easy-rsa-master/kubelet
    mkdir easy-rsa-master/aggregator
    cp -r easy-rsa-master/easyrsa3/* easy-rsa-master/aggregator) &>${cert_create_debug_output} || true
  CERT_DIR="${KUBE_TEMP}/easy-rsa-master/easyrsa3"
  AGGREGATOR_CERT_DIR="${KUBE_TEMP}/easy-rsa-master/aggregator"
  if [ ! -x "${CERT_DIR}/easyrsa" -o ! -x "${AGGREGATOR_CERT_DIR}/easyrsa" ]; then
    # TODO(roberthbailey,porridge): add better error handling here,
    # see https://github.com/kubernetes/kubernetes/issues/55229
    cat "${cert_create_debug_output}" >&2
    echo "=== Failed to setup easy-rsa: Aborting ===" >&2
    exit 2
  fi
}

# Runs the easy RSA commands to generate certificate files.
# The generated files are IN ${CERT_DIR}
#
# Assumed vars
#   KUBE_TEMP
#   MASTER_NAME
#   CERT_DIR
#   PRIMARY_CN: Primary canonical name
#   SANS: Subject alternate names
#
#
function generate-certs {
  local -r cert_create_debug_output=$(mktemp "${KUBE_TEMP}/cert_create_debug_output.XXX")
  # Note: This was heavily cribbed from make-ca-cert.sh
  (set -x
    cd "${CERT_DIR}"
    ./easyrsa init-pki
    # this puts the cert into pki/ca.crt and the key into pki/private/ca.key
    ./easyrsa --batch "--req-cn=${PRIMARY_CN}@$(date +%s)" build-ca nopass
    ./easyrsa --subject-alt-name="${SANS}" build-server-full "${MASTER_NAME}" nopass
    ./easyrsa build-client-full kube-apiserver nopass

    kube::util::ensure-cfssl "${KUBE_TEMP}/cfssl"

    # make the config for the signer
    echo '{"signing":{"default":{"expiry":"43800h","usages":["signing","key encipherment","client auth"]}}}' > "ca-config.json"
    # create the kubelet client cert with the correct groups
    echo '{"CN":"kubelet","names":[{"O":"system:nodes"}],"hosts":[""],"key":{"algo":"rsa","size":2048}}' | "${CFSSL_BIN}" gencert -ca=pki/ca.crt -ca-key=pki/private/ca.key -config=ca-config.json - | "${CFSSLJSON_BIN}" -bare kubelet
    mv "kubelet-key.pem" "pki/private/kubelet.key"
    mv "kubelet.pem" "pki/issued/kubelet.crt"
    rm -f "kubelet.csr"

    # Make a superuser client cert with subject "O=system:masters, CN=kubecfg"
    ./easyrsa --dn-mode=org \
      --req-cn=kubecfg --req-org=system:masters \
      --req-c= --req-st= --req-city= --req-email= --req-ou= \
      build-client-full kubecfg nopass) &>${cert_create_debug_output} || true
  local output_file_missing=0
  local output_file
  for output_file in \
    "${CERT_DIR}/pki/private/ca.key" \
    "${CERT_DIR}/pki/ca.crt" \
    "${CERT_DIR}/pki/issued/${MASTER_NAME}.crt" \
    "${CERT_DIR}/pki/private/${MASTER_NAME}.key" \
    "${CERT_DIR}/pki/issued/kubelet.crt" \
    "${CERT_DIR}/pki/private/kubelet.key" \
    "${CERT_DIR}/pki/issued/kubecfg.crt" \
    "${CERT_DIR}/pki/private/kubecfg.key" \
    "${CERT_DIR}/pki/issued/kube-apiserver.crt" \
    "${CERT_DIR}/pki/private/kube-apiserver.key"
  do
    if [[ ! -s "${output_file}" ]]; then
      echo "Expected file ${output_file} not created" >&2
      output_file_missing=1
    fi
  done
  if (( $output_file_missing )); then
    # TODO(roberthbailey,porridge): add better error handling here,
    # see https://github.com/kubernetes/kubernetes/issues/55229
    cat "${cert_create_debug_output}" >&2
    echo "=== Failed to generate master certificates: Aborting ===" >&2
    exit 2
  fi
}

# Runs the easy RSA commands to generate aggregator certificate files.
# The generated files are in ${AGGREGATOR_CERT_DIR}
#
# Assumed vars
#   KUBE_TEMP
#   AGGREGATOR_MASTER_NAME
#   AGGREGATOR_CERT_DIR
#   AGGREGATOR_PRIMARY_CN: Primary canonical name
#   AGGREGATOR_SANS: Subject alternate names
#
#
function generate-aggregator-certs {
  local -r cert_create_debug_output=$(mktemp "${KUBE_TEMP}/cert_create_debug_output.XXX")
  # Note: This was heavily cribbed from make-ca-cert.sh
  (set -x
    cd "${KUBE_TEMP}/easy-rsa-master/aggregator"
    ./easyrsa init-pki
    # this puts the cert into pki/ca.crt and the key into pki/private/ca.key
    ./easyrsa --batch "--req-cn=${AGGREGATOR_PRIMARY_CN}@$(date +%s)" build-ca nopass
    ./easyrsa --subject-alt-name="${AGGREGATOR_SANS}" build-server-full "${AGGREGATOR_MASTER_NAME}" nopass
    ./easyrsa build-client-full aggregator-apiserver nopass

    kube::util::ensure-cfssl "${KUBE_TEMP}/cfssl"

    # make the config for the signer
    echo '{"signing":{"default":{"expiry":"43800h","usages":["signing","key encipherment","client auth"]}}}' > "ca-config.json"
    # create the aggregator client cert with the correct groups
    echo '{"CN":"aggregator","hosts":[""],"key":{"algo":"rsa","size":2048}}' | "${CFSSL_BIN}" gencert -ca=pki/ca.crt -ca-key=pki/private/ca.key -config=ca-config.json - | "${CFSSLJSON_BIN}" -bare proxy-client
    mv "proxy-client-key.pem" "pki/private/proxy-client.key"
    mv "proxy-client.pem" "pki/issued/proxy-client.crt"
    rm -f "proxy-client.csr"

    # Make a superuser client cert with subject "O=system:masters, CN=kubecfg"
    ./easyrsa --dn-mode=org \
      --req-cn=proxy-clientcfg --req-org=system:aggregator \
      --req-c= --req-st= --req-city= --req-email= --req-ou= \
      build-client-full proxy-clientcfg nopass) &>${cert_create_debug_output} || true
  local output_file_missing=0
  local output_file
  for output_file in \
    "${AGGREGATOR_CERT_DIR}/pki/private/ca.key" \
    "${AGGREGATOR_CERT_DIR}/pki/ca.crt" \
    "${AGGREGATOR_CERT_DIR}/pki/issued/proxy-client.crt" \
    "${AGGREGATOR_CERT_DIR}/pki/private/proxy-client.key"
  do
    if [[ ! -s "${output_file}" ]]; then
      echo "Expected file ${output_file} not created" >&2
      output_file_missing=1
    fi
  done
  if (( $output_file_missing )); then
    # TODO(roberthbailey,porridge): add better error handling here,
    # see https://github.com/kubernetes/kubernetes/issues/55229
    cat "${cert_create_debug_output}" >&2
    echo "=== Failed to generate aggregator certificates: Aborting ===" >&2
    exit 2
  fi
}

#
# Using provided master env, extracts value from provided key.
#
# Args:
# $1 master env (kube-env of master; result of calling get-master-env)
# $2 env key to use
function get-env-val() {
  local match=`(echo "${1}" | grep -E "^${2}:") || echo ""`
  if [[ -z ${match} ]]; then
    echo ""
  fi
  echo ${match} | cut -d : -f 2 | cut -d \' -f 2
}

# Load the master env by calling get-master-env, and extract important values
function parse-master-env() {
  # Get required master env vars
  local master_env=$(get-master-env)
  KUBE_PROXY_TOKEN=$(get-env-val "${master_env}" "KUBE_PROXY_TOKEN")
  NODE_PROBLEM_DETECTOR_TOKEN=$(get-env-val "${master_env}" "NODE_PROBLEM_DETECTOR_TOKEN")
  CA_CERT_BASE64=$(get-env-val "${master_env}" "CA_CERT")
  CA_KEY_BASE64=$(get-env-val "${master_env}" "CA_KEY")
  KUBEAPISERVER_CERT_BASE64=$(get-env-val "${master_env}" "KUBEAPISERVER_CERT")
  KUBEAPISERVER_KEY_BASE64=$(get-env-val "${master_env}" "KUBEAPISERVER_KEY")
  EXTRA_DOCKER_OPTS=$(get-env-val "${master_env}" "EXTRA_DOCKER_OPTS")
  KUBELET_CERT_BASE64=$(get-env-val "${master_env}" "KUBELET_CERT")
  KUBELET_KEY_BASE64=$(get-env-val "${master_env}" "KUBELET_KEY")
  MASTER_CERT_BASE64=$(get-env-val "${master_env}" "MASTER_CERT")
  MASTER_KEY_BASE64=$(get-env-val "${master_env}" "MASTER_KEY")
  AGGREGATOR_CA_KEY_BASE64=$(get-env-val "${master_env}" "AGGREGATOR_CA_KEY")
  REQUESTHEADER_CA_CERT_BASE64=$(get-env-val "${master_env}" "REQUESTHEADER_CA_CERT")
  PROXY_CLIENT_CERT_BASE64=$(get-env-val "${master_env}" "PROXY_CLIENT_CERT")
  PROXY_CLIENT_KEY_BASE64=$(get-env-val "${master_env}" "PROXY_CLIENT_KEY")
  ENABLE_LEGACY_ABAC=$(get-env-val "${master_env}" "ENABLE_LEGACY_ABAC")
}

# Update or verify required gcloud components are installed
# at minimum required version.
# Assumed vars
#   KUBE_PROMPT_FOR_UPDATE
function update-or-verify-gcloud() {
  local sudo_prefix=""
  if [ ! -w $(dirname `which gcloud`) ]; then
    sudo_prefix="sudo"
  fi
  # update and install components as needed
  if [[ "${KUBE_PROMPT_FOR_UPDATE}" == "y" ]]; then
    ${sudo_prefix} gcloud ${gcloud_prompt:-} components install alpha
    ${sudo_prefix} gcloud ${gcloud_prompt:-} components install beta
    ${sudo_prefix} gcloud ${gcloud_prompt:-} components update
  else
    local version=$(gcloud version --format=json)
    python -c'
import json,sys
from distutils import version

minVersion = version.LooseVersion("1.3.0")
required = [ "alpha", "beta", "core" ]
data = json.loads(sys.argv[1])
rel = data.get("Google Cloud SDK")
if rel != "HEAD" and version.LooseVersion(rel) < minVersion:
  print("gcloud version out of date ( < %s )" % minVersion)
  exit(1)
missing = []
for c in required:
  if not data.get(c):
    missing += [c]
if missing:
  for c in missing:
    print ("missing required gcloud component \"{0}\"".format(c))
  exit(1)
    ' """${version}"""
  fi
}

# Robustly try to create a static ip.
# $1: The name of the ip to create
# $2: The name of the region to create the ip in.
function create-static-ip() {
  detect-project
  local attempt=0
  local REGION="$2"
  while true; do
    if gcloud compute addresses create "$1" \
      --project "${PROJECT}" \
      --region "${REGION}" -q > /dev/null; then
      # successful operation - wait until it's visible
      start="$(date +%s)"
      while true; do
        now="$(date +%s)"
        # Timeout set to 15 minutes
        if [[ $((now - start)) -gt 900 ]]; then
          echo "Timeout while waiting for master IP visibility"
          exit 2
        fi
        if gcloud compute addresses describe "$1" --project "${PROJECT}" --region "${REGION}" >/dev/null 2>&1; then
          break
        fi
        echo "Master IP not visible yet. Waiting..."
        sleep 5
      done
      break
    fi

    if gcloud compute addresses describe "$1" \
      --project "${PROJECT}" \
      --region "${REGION}" >/dev/null 2>&1; then
      # it exists - postcondition satisfied
      break
    fi

    if (( attempt > 4 )); then
      echo -e "${color_red}Failed to create static ip $1 ${color_norm}" >&2
      exit 2
    fi
    attempt=$(($attempt+1))
    echo -e "${color_yellow}Attempt $attempt failed to create static ip $1. Retrying.${color_norm}" >&2
    sleep $(($attempt * 5))
  done
}

# Robustly try to create a firewall rule.
# $1: The name of firewall rule.
# $2: IP ranges.
# $3: Target tags for this firewall rule.
function create-firewall-rule() {
  detect-project
  local attempt=0
  while true; do
    if ! gcloud compute firewall-rules create "$1" \
      --project "${NETWORK_PROJECT}" \
      --network "${NETWORK}" \
      --source-ranges "$2" \
      --target-tags "$3" \
      --allow tcp,udp,icmp,esp,ah,sctp; then
      if (( attempt > 4 )); then
        echo -e "${color_red}Failed to create firewall rule $1 ${color_norm}" >&2
        exit 2
      fi
      echo -e "${color_yellow}Attempt $(($attempt+1)) failed to create firewall rule $1. Retrying.${color_norm}" >&2
      attempt=$(($attempt+1))
      sleep $(($attempt * 5))
    else
        break
    fi
  done
}

# Format the string argument for gcloud network.
function make-gcloud-network-argument() {
  local network_project="$1"
  local region="$2"
  local network="$3"
  local subnet="$4"
  local address="$5"          # optional
  local enable_ip_alias="$6"  # optional
  local alias_size="$7"       # optional

  local networkURL="projects/${network_project}/global/networks/${network}"
  local subnetURL="projects/${network_project}/regions/${region}/subnetworks/${subnet:-}"

  local ret=""

  if [[ "${enable_ip_alias}" == 'true' ]]; then
    ret="--network-interface"
    ret="${ret} network=${networkURL}"
    # If address is omitted, instance will not receive an external IP.
    ret="${ret},address=${address:-}"
    ret="${ret},subnet=${subnetURL}"
    ret="${ret},aliases=pods-default:${alias_size}"
    ret="${ret} --no-can-ip-forward"
  else
    if [[ -n ${subnet:-} ]]; then
      ret="${ret} --subnet ${subnetURL}"
    else
      ret="${ret} --network ${networkURL}"
    fi

    ret="${ret} --can-ip-forward"
    if [[ -n ${address:-} ]]; then
      ret="${ret} --address ${address}"
    fi
  fi

  echo "${ret}"
}

# $1: version (required)
function get-template-name-from-version() {
  # trim template name to pass gce name validation
  echo "${NODE_INSTANCE_PREFIX}-template-${1}" | cut -c 1-63 | sed 's/[\.\+]/-/g;s/-*$//g'
}

# validates the NODE_LOCAL_SSDS_EXT variable
function validate-node-local-ssds-ext(){
  ssdopts="${1}"

  if [[ -z "${ssdopts[0]}" || -z "${ssdopts[1]}" || -z "${ssdopts[2]}" ]]; then
	  echo -e "${color_red}Local SSD: NODE_LOCAL_SSDS_EXT is malformed, found ${ssdopts[0]-_},${ssdopts[1]-_},${ssdopts[2]-_} ${color_norm}" >&2
    exit 2
  fi
  if [[ "${ssdopts[1]}" != "scsi" && "${ssdopts[1]}" != "nvme" ]]; then
    echo -e "${color_red}Local SSD: Interface must be scsi or nvme, found: ${ssdopts[1]} ${color_norm}" >&2
    exit 2
  fi
  if [[ "${ssdopts[2]}" != "fs" && "${ssdopts[2]}" != "block" ]]; then
    echo -e "${color_red}Local SSD: Filesystem type must be fs or block, found: ${ssdopts[2]} ${color_norm}"  >&2
    exit 2
  fi
  local_ssd_ext_count=$((local_ssd_ext_count+ssdopts[0]))
  if [[ "${local_ssd_ext_count}" -gt "${GCE_MAX_LOCAL_SSD}" || "${local_ssd_ext_count}" -lt 1 ]]; then
    echo -e "${color_red}Local SSD: Total number of local ssds must range from 1 to 8, found: ${local_ssd_ext_count} ${color_norm}" >&2
    exit 2
  fi
}

# Robustly try to create an instance template.
# $1: The name of the instance template.
# $2: The scopes flag.
# $3: String of comma-separated metadata entries (must all be from a file).
function create-node-template() {
  detect-project
  detect-subnetworks
  local template_name="$1"

  # First, ensure the template doesn't exist.
  # TODO(zmerlynn): To make this really robust, we need to parse the output and
  #                 add retries. Just relying on a non-zero exit code doesn't
  #                 distinguish an ephemeral failed call from a "not-exists".
  if gcloud compute instance-templates describe "$template_name" --project "${PROJECT}" &>/dev/null; then
    echo "Instance template ${1} already exists; deleting." >&2
    if ! gcloud compute instance-templates delete "$template_name" --project "${PROJECT}" --quiet &>/dev/null; then
      echo -e "${color_yellow}Failed to delete existing instance template${color_norm}" >&2
      exit 2
    fi
  fi

  local gcloud="gcloud"

  local accelerator_args=""
  # VMs with Accelerators cannot be live migrated.
  # More details here - https://cloud.google.com/compute/docs/gpus/add-gpus#create-new-gpu-instance
  if [[ ! -z "${NODE_ACCELERATORS}" ]]; then
    accelerator_args="--maintenance-policy TERMINATE --restart-on-failure --accelerator ${NODE_ACCELERATORS}"
    gcloud="gcloud beta"
  fi

  if [[ "${ENABLE_IP_ALIASES:-}" == 'true' ]]; then
    gcloud="gcloud beta"
  fi

  local preemptible_minions=""
  if [[ "${PREEMPTIBLE_NODE}" == "true" ]]; then
    preemptible_minions="--preemptible --maintenance-policy TERMINATE"
  fi

  local local_ssds=""
  local_ssd_ext_count=0
  if [[ ! -z ${NODE_LOCAL_SSDS_EXT:-} ]]; then
    IFS=";" read -r -a ssdgroups <<< "${NODE_LOCAL_SSDS_EXT:-}"
    for ssdgroup in "${ssdgroups[@]}"
    do
      IFS="," read -r -a ssdopts <<< "${ssdgroup}"
      validate-node-local-ssds-ext "${ssdopts}"
      for i in $(seq ${ssdopts[0]}); do
        local_ssds="$local_ssds--local-ssd=interface=${ssdopts[1]} "
      done
    done
  fi

  if [[ ! -z ${NODE_LOCAL_SSDS+x} ]]; then
    # The NODE_LOCAL_SSDS check below fixes issue #49171
    # Some versions of seq will count down from 1 if "seq 0" is specified
    if [[ ${NODE_LOCAL_SSDS} -ge 1 ]]; then
      for i in $(seq ${NODE_LOCAL_SSDS}); do
        local_ssds="$local_ssds--local-ssd=interface=SCSI "
      done
    fi
  fi


  local network=$(make-gcloud-network-argument \
    "${NETWORK_PROJECT}" \
    "${REGION}" \
    "${NETWORK}" \
    "${SUBNETWORK:-}" \
    "" \
    "${ENABLE_IP_ALIASES:-}" \
    "${IP_ALIAS_SIZE:-}")

  local attempt=1
  while true; do
    echo "Attempt ${attempt} to create ${1}" >&2
    if ! ${gcloud} compute instance-templates create \
      "$template_name" \
      --project "${PROJECT}" \
      --machine-type "${NODE_SIZE}" \
      --boot-disk-type "${NODE_DISK_TYPE}" \
      --boot-disk-size "${NODE_DISK_SIZE}" \
      --image-project="${NODE_IMAGE_PROJECT}" \
      --image "${NODE_IMAGE}" \
      --service-account "${NODE_SERVICE_ACCOUNT}" \
      --tags "${NODE_TAG}" \
      ${accelerator_args} \
      ${local_ssds} \
      --region "${REGION}" \
      ${network} \
      ${preemptible_minions} \
      $2 \
      --metadata-from-file $3 >&2; then
        if (( attempt > 5 )); then
          echo -e "${color_red}Failed to create instance template $template_name ${color_norm}" >&2
          exit 2
        fi
        echo -e "${color_yellow}Attempt ${attempt} failed to create instance template $template_name. Retrying.${color_norm}" >&2
        attempt=$(($attempt+1))
        sleep $(($attempt * 5))

        # In case the previous attempt failed with something like a
        # Backend Error and left the entry laying around, delete it
        # before we try again.
        gcloud compute instance-templates delete "$template_name" --project "${PROJECT}" &>/dev/null || true
    else
        break
    fi
  done
}

# Instantiate a kubernetes cluster
#
# Assumed vars
#   KUBE_ROOT
#   <Various vars set in config file>
function kube-up() {
  kube::util::ensure-temp-dir
  detect-project

  load-or-gen-kube-basicauth
  load-or-gen-kube-bearertoken

  # Make sure we have the tar files staged on Google Storage
  find-release-tars
  upload-server-tars

  # ensure that environmental variables specifying number of migs to create
  set_num_migs

  if [[ ${KUBE_USE_EXISTING_MASTER:-} == "true" ]]; then
    detect-master
    parse-master-env
    create-subnetworks
    detect-subnetworks
    create-nodes
  elif [[ ${KUBE_REPLICATE_EXISTING_MASTER:-} == "true" ]]; then
    if  [[ "${MASTER_OS_DISTRIBUTION}" != "gci" && "${MASTER_OS_DISTRIBUTION}" != "ubuntu" ]]; then
      echo "Master replication supported only for gci and ubuntu"
      return 1
    fi
    create-loadbalancer
    # If replication of master fails, we need to ensure that the replica is removed from etcd clusters.
    if ! replicate-master; then
      remove-replica-from-etcd 2379 || true
      remove-replica-from-etcd 4002 || true
    fi
  else
    check-existing
    create-network
    create-subnetworks
    detect-subnetworks
    write-cluster-location
    write-cluster-name
    create-autoscaler-config
    create-master
    create-nodes-firewall
    create-nodes-template
    create-nodes
    check-cluster
  fi
}

function check-existing() {
  local running_in_terminal=false
  # May be false if tty is not allocated (for example with ssh -T).
  if [[ -t 1 ]]; then
    running_in_terminal=true
  fi

  if [[ ${running_in_terminal} == "true" || ${KUBE_UP_AUTOMATIC_CLEANUP} == "true" ]]; then
    if ! check-resources; then
      local run_kube_down="n"
      echo "${KUBE_RESOURCE_FOUND} found." >&2
      # Get user input only if running in terminal.
      if [[ ${running_in_terminal} == "true" && ${KUBE_UP_AUTOMATIC_CLEANUP} == "false" ]]; then
        read -p "Would you like to shut down the old cluster (call kube-down)? [y/N] " run_kube_down
      fi
      if [[ ${run_kube_down} == "y" || ${run_kube_down} == "Y" || ${KUBE_UP_AUTOMATIC_CLEANUP} == "true" ]]; then
        echo "... calling kube-down" >&2
        kube-down
      fi
    fi
  fi
}

# TODO(#54017): Remove below logics for handling deprecated network mode field.
# `x_gcloud_mode` was replaced by `x_gcloud_subnet_mode` in gcloud 175.0.0 and
# the content changed as well. Keeping such logic to make the transition eaiser.
function check-network-mode() {
  local mode="$(gcloud compute networks list --filter="name=('${NETWORK}')" --project ${NETWORK_PROJECT} --format='value(x_gcloud_subnet_mode)' || true)"
  if [[ -z "${mode}" ]]; then
    mode="$(gcloud compute networks list --filter="name=('${NETWORK}')" --project ${NETWORK_PROJECT} --format='value(x_gcloud_mode)' || true)"
  fi
  # The deprecated field uses lower case. Convert to upper case for consistency.
  echo "$(echo $mode | tr [a-z] [A-Z])"
}

function create-network() {
  if ! gcloud compute networks --project "${NETWORK_PROJECT}" describe "${NETWORK}" &>/dev/null; then
    # The network needs to be created synchronously or we have a race. The
    # firewalls can be added concurrent with instance creation.
    local network_mode="auto"
    if [[ "${CREATE_CUSTOM_NETWORK:-}" == "true" ]]; then
      network_mode="custom"
    fi
    echo "Creating new ${network_mode} network: ${NETWORK}"
    gcloud compute networks create --project "${NETWORK_PROJECT}" "${NETWORK}" --subnet-mode="${network_mode}"
  else
    PREEXISTING_NETWORK=true
    PREEXISTING_NETWORK_MODE="$(check-network-mode)"
    echo "Found existing network ${NETWORK} in ${PREEXISTING_NETWORK_MODE} mode."
  fi

  if ! gcloud compute firewall-rules --project "${NETWORK_PROJECT}" describe "${CLUSTER_NAME}-default-internal-master" &>/dev/null; then
    gcloud compute firewall-rules create "${CLUSTER_NAME}-default-internal-master" \
      --project "${NETWORK_PROJECT}" \
      --network "${NETWORK}" \
      --source-ranges "10.0.0.0/8" \
      --allow "tcp:1-2379,tcp:2382-65535,udp:1-65535,icmp" \
      --target-tags "${MASTER_TAG}"&
  fi

  if ! gcloud compute firewall-rules --project "${NETWORK_PROJECT}" describe "${CLUSTER_NAME}-default-internal-node" &>/dev/null; then
    gcloud compute firewall-rules create "${CLUSTER_NAME}-default-internal-node" \
      --project "${NETWORK_PROJECT}" \
      --network "${NETWORK}" \
      --source-ranges "10.0.0.0/8" \
      --allow "tcp:1-65535,udp:1-65535,icmp" \
      --target-tags "${NODE_TAG}"&
  fi

  if ! gcloud compute firewall-rules describe --project "${NETWORK_PROJECT}" "${NETWORK}-default-ssh" &>/dev/null; then
    gcloud compute firewall-rules create "${NETWORK}-default-ssh" \
      --project "${NETWORK_PROJECT}" \
      --network "${NETWORK}" \
      --source-ranges "0.0.0.0/0" \
      --allow "tcp:22" &
  fi
}

function expand-default-subnetwork() {
  gcloud compute networks update "${NETWORK}" \
    --switch-to-custom-subnet-mode \
    --project "${NETWORK_PROJECT}" \
    --quiet || true
  gcloud compute networks subnets expand-ip-range "${NETWORK}" \
    --region="${REGION}" \
    --project "${NETWORK_PROJECT}" \
    --prefix-length=19 \
    --quiet
}

function create-subnetworks() {
  case ${ENABLE_IP_ALIASES} in
    true) echo "IP aliases are enabled. Creating subnetworks.";;
    false)
      echo "IP aliases are disabled."
      if [[ "${ENABLE_BIG_CLUSTER_SUBNETS}" = "true" ]]; then
        if [[  "${PREEXISTING_NETWORK}" != "true" ]]; then
          expand-default-subnetwork
        else
          echo "${color_yellow}Using pre-existing network ${NETWORK}, subnets won't be expanded to /19!${color_norm}"
        fi
      elif [[ "${CREATE_CUSTOM_NETWORK:-}" == "true" && "${PREEXISTING_NETWORK}" != "true" ]]; then
          gcloud compute networks subnets create "${SUBNETWORK}" --project "${NETWORK_PROJECT}" --region "${REGION}" --network "${NETWORK}" --range "${NODE_IP_RANGE}"
      fi
      return;;
    *) echo "${color_red}Invalid argument to ENABLE_IP_ALIASES${color_norm}"
       exit 1;;
  esac

  # Look for the alias subnet, it must exist and have a secondary
  # range configured.
  local subnet=$(gcloud beta compute networks subnets describe \
    --project "${NETWORK_PROJECT}" \
    --region ${REGION} \
    ${IP_ALIAS_SUBNETWORK} 2>/dev/null)
  if [[ -z ${subnet} ]]; then
    echo "Creating subnet ${NETWORK}:${IP_ALIAS_SUBNETWORK}"
    gcloud beta compute networks subnets create \
      ${IP_ALIAS_SUBNETWORK} \
      --description "Automatically generated subnet for ${INSTANCE_PREFIX} cluster. This will be removed on cluster teardown." \
      --project "${NETWORK_PROJECT}" \
      --network ${NETWORK} \
      --region ${REGION} \
      --range ${NODE_IP_RANGE} \
      --secondary-range "pods-default=${CLUSTER_IP_RANGE}" \
      --secondary-range "services-default=${SERVICE_CLUSTER_IP_RANGE}"
    echo "Created subnetwork ${IP_ALIAS_SUBNETWORK}"
  else
    if ! echo ${subnet} | grep --quiet secondaryIpRanges ${subnet}; then
      echo "${color_red}Subnet ${IP_ALIAS_SUBNETWORK} does not have a secondary range${color_norm}"
      exit 1
    fi
  fi
}

# detect-subnetworks sets the SUBNETWORK var if not already set
# Assumed vars:
#   NETWORK
#   REGION
#   NETWORK_PROJECT
#
# Optional vars:
#   SUBNETWORK
#   IP_ALIAS_SUBNETWORK
function detect-subnetworks() {
  if [[ -n ${SUBNETWORK:-} ]]; then
    echo "Using subnet ${SUBNETWORK}"
    return 0
  fi

  if [[ -n ${IP_ALIAS_SUBNETWORK:-} ]]; then
    SUBNETWORK=${IP_ALIAS_SUBNETWORK}
    echo "Using IP Alias subnet ${SUBNETWORK}"
    return 0
  fi

  SUBNETWORK=$(gcloud beta compute networks subnets list \
    --network=${NETWORK} \
    --regions=${REGION} \
    --project=${NETWORK_PROJECT} \
    --limit=1 \
    --format='value(name)' 2>/dev/null)

  if [[ -n ${SUBNETWORK:-} ]]; then
    echo "Found subnet for region ${REGION} in network ${NETWORK}: ${SUBNETWORK}"
    return 0
  fi

  echo "${color_red}Could not find subnetwork with region ${REGION}, network ${NETWORK}, and project ${NETWORK_PROJECT}"
}

function delete-all-firewall-rules() {
  if fws=$(gcloud compute firewall-rules list --project "${NETWORK_PROJECT}" --filter="network=${NETWORK}" --format="value(name)"); then
    echo "Deleting firewall rules remaining in network ${NETWORK}: ${fws}"
    delete-firewall-rules "$fws"
  else
    echo "Failed to list firewall rules from the network ${NETWORK}"
  fi
}

function delete-firewall-rules() {
  for fw in $@; do
    if [[ -n $(gcloud compute firewall-rules --project "${NETWORK_PROJECT}" describe "${fw}" --format='value(name)' 2>/dev/null || true) ]]; then
      gcloud compute firewall-rules delete --project "${NETWORK_PROJECT}" --quiet "${fw}" &
    fi
  done
  kube::util::wait-for-jobs || {
    echo -e "${color_red}Failed to delete firewall rules.${color_norm}" >&2
  }
}

function delete-network() {
  if [[ -n $(gcloud compute networks --project "${NETWORK_PROJECT}" describe "${NETWORK}" --format='value(name)' 2>/dev/null || true) ]]; then
    if ! gcloud compute networks delete --project "${NETWORK_PROJECT}" --quiet "${NETWORK}"; then
      echo "Failed to delete network '${NETWORK}'. Listing firewall-rules:"
      gcloud compute firewall-rules --project "${NETWORK_PROJECT}" list --filter="network=${NETWORK}"
      return 1
    fi
  fi
}

function delete-subnetworks() {
  # If running in custom mode network we need to delete subnets manually.
  mode="$(check-network-mode)"
  if [[ "${mode}" == "CUSTOM" ]]; then
    if [[ "${ENABLE_BIG_CLUSTER_SUBNETS}" = "true" ]]; then
      echo "Deleting default subnets..."
      # This value should be kept in sync with number of regions.
      local parallelism=9
      gcloud compute networks subnets list --network="${NETWORK}" --project "${NETWORK_PROJECT}" --format='value(region.basename())' | \
        xargs -i -P ${parallelism} gcloud --quiet compute networks subnets delete "${NETWORK}" --project "${NETWORK_PROJECT}" --region="{}" || true
    elif [[ "${CREATE_CUSTOM_NETWORK:-}" == "true" ]]; then
      echo "Deleting custom subnet..."
      gcloud --quiet compute networks subnets delete "${SUBNETWORK}" --project "${NETWORK_PROJECT}" --region="${REGION}" || true
    fi
    return
  fi

  # If we reached here, it means we're not using custom network.
  # So the only thing we need to check is if IP-aliases was turned
  # on and we created a subnet for it. If so, we should delete it.
  if [[ ${ENABLE_IP_ALIASES:-} == "true" ]]; then
    # Only delete the subnet if we created it (i.e it's not pre-existing).
    if [[ -z "${KUBE_GCE_IP_ALIAS_SUBNETWORK:-}" ]]; then
      echo "Removing auto-created subnet ${NETWORK}:${IP_ALIAS_SUBNETWORK}"
      if [[ -n $(gcloud beta compute networks subnets describe \
            --project "${NETWORK_PROJECT}" \
            --region ${REGION} \
            ${IP_ALIAS_SUBNETWORK} 2>/dev/null) ]]; then
        gcloud beta --quiet compute networks subnets delete \
          --project "${NETWORK_PROJECT}" \
          --region ${REGION} \
          ${IP_ALIAS_SUBNETWORK}
      fi
    fi
  fi
}

# Generates SSL certificates for etcd cluster. Uses cfssl program.
#
# Assumed vars:
#   KUBE_TEMP: temporary directory
#   NUM_NODES: #nodes in the cluster
#
# Args:
#  $1: host name
#  $2: CA certificate
#  $3: CA key
#
# If CA cert/key is empty, the function will also generate certs for CA.
#
# Vars set:
#   ETCD_CA_KEY_BASE64
#   ETCD_CA_CERT_BASE64
#   ETCD_PEER_KEY_BASE64
#   ETCD_PEER_CERT_BASE64
#
function create-etcd-certs {
  local host=${1}
  local ca_cert=${2:-}
  local ca_key=${3:-}

  GEN_ETCD_CA_CERT="${ca_cert}" GEN_ETCD_CA_KEY="${ca_key}" \
    generate-etcd-cert "${KUBE_TEMP}/cfssl" "${host}" "peer" "peer"

  pushd "${KUBE_TEMP}/cfssl"
  ETCD_CA_KEY_BASE64=$(cat "ca-key.pem" | base64 | tr -d '\r\n')
  ETCD_CA_CERT_BASE64=$(cat "ca.pem" | gzip | base64 | tr -d '\r\n')
  ETCD_PEER_KEY_BASE64=$(cat "peer-key.pem" | base64 | tr -d '\r\n')
  ETCD_PEER_CERT_BASE64=$(cat "peer.pem" | gzip | base64 | tr -d '\r\n')
  popd
}

function create-master() {
  echo "Starting master and configuring firewalls"
  gcloud compute firewall-rules create "${MASTER_NAME}-https" \
    --project "${NETWORK_PROJECT}" \
    --network "${NETWORK}" \
    --target-tags "${MASTER_TAG}" \
    --allow tcp:443 &

  # We have to make sure the disk is created before creating the master VM, so
  # run this in the foreground.
  gcloud compute disks create "${MASTER_NAME}-pd" \
    --project "${PROJECT}" \
    --zone "${ZONE}" \
    --type "${MASTER_DISK_TYPE}" \
    --size "${MASTER_DISK_SIZE}"

  # Create rule for accessing and securing etcd servers.
  if ! gcloud compute firewall-rules --project "${NETWORK_PROJECT}" describe "${MASTER_NAME}-etcd" &>/dev/null; then
    gcloud compute firewall-rules create "${MASTER_NAME}-etcd" \
      --project "${NETWORK_PROJECT}" \
      --network "${NETWORK}" \
      --source-tags "${MASTER_TAG}" \
      --allow "tcp:2380,tcp:2381" \
      --target-tags "${MASTER_TAG}" &
  fi

  # Generate a bearer token for this cluster. We push this separately
  # from the other cluster variables so that the client (this
  # computer) can forget it later. This should disappear with
  # http://issue.k8s.io/3168
  KUBE_PROXY_TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
  if [[ "${ENABLE_NODE_PROBLEM_DETECTOR:-}" == "standalone" ]]; then
    NODE_PROBLEM_DETECTOR_TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
  fi

  # Reserve the master's IP so that it can later be transferred to another VM
  # without disrupting the kubelets.
  create-static-ip "${MASTER_NAME}-ip" "${REGION}"
  MASTER_RESERVED_IP=$(gcloud compute addresses describe "${MASTER_NAME}-ip" \
    --project "${PROJECT}" --region "${REGION}" -q --format='value(address)')

  if [[ "${REGISTER_MASTER_KUBELET:-}" == "true" ]]; then
    KUBELET_APISERVER="${MASTER_RESERVED_IP}"
  fi

  KUBERNETES_MASTER_NAME="${MASTER_RESERVED_IP}"
  MASTER_ADVERTISE_ADDRESS="${MASTER_RESERVED_IP}"

  create-certs "${MASTER_RESERVED_IP}"
  create-etcd-certs ${MASTER_NAME}

  if [[ "${NUM_NODES}" -ge "50" ]]; then
    # We block on master creation for large clusters to avoid doing too much
    # unnecessary work in case master start-up fails (like creation of nodes).
    create-master-instance "${MASTER_RESERVED_IP}"
  else
    create-master-instance "${MASTER_RESERVED_IP}" &
  fi
}

# Adds master replica to etcd cluster.
#
# Assumed vars:
#   REPLICA_NAME
#   PROJECT
#   EXISTING_MASTER_NAME
#   EXISTING_MASTER_ZONE
#
# $1: etcd client port
# $2: etcd internal port
# returns the result of ssh command which adds replica
function add-replica-to-etcd() {
  local -r client_port="${1}"
  local -r internal_port="${2}"
  gcloud compute ssh "${EXISTING_MASTER_NAME}" \
    --project "${PROJECT}" \
    --zone "${EXISTING_MASTER_ZONE}" \
    --command \
      "curl localhost:${client_port}/v2/members -XPOST -H \"Content-Type: application/json\" -d '{\"peerURLs\":[\"https://${REPLICA_NAME}:${internal_port}\"]}' -s"
  return $?
}

# Sets EXISTING_MASTER_NAME and EXISTING_MASTER_ZONE variables.
#
# Assumed vars:
#   PROJECT
#
# NOTE: Must be in sync with get-replica-name-regexp
function set-existing-master() {
  local existing_master=$(gcloud compute instances list \
    --project "${PROJECT}" \
    --filter "name ~ '$(get-replica-name-regexp)'" \
    --format "value(name,zone)" | head -n1)
  EXISTING_MASTER_NAME="$(echo "${existing_master}" | cut -f1)"
  EXISTING_MASTER_ZONE="$(echo "${existing_master}" | cut -f2)"
}

function replicate-master() {
  set-replica-name
  set-existing-master

  echo "Experimental: replicating existing master ${EXISTING_MASTER_ZONE}/${EXISTING_MASTER_NAME} as ${ZONE}/${REPLICA_NAME}"

  # Before we do anything else, we should configure etcd to expect more replicas.
  if ! add-replica-to-etcd 2379 2380; then
    echo "Failed to add master replica to etcd cluster."
    return 1
  fi
  if ! add-replica-to-etcd 4002 2381; then
    echo "Failed to add master replica to etcd events cluster."
    return 1
  fi

  # We have to make sure the disk is created before creating the master VM, so
  # run this in the foreground.
  gcloud compute disks create "${REPLICA_NAME}-pd" \
    --project "${PROJECT}" \
    --zone "${ZONE}" \
    --type "${MASTER_DISK_TYPE}" \
    --size "${MASTER_DISK_SIZE}"

  local existing_master_replicas="$(get-all-replica-names)"
  replicate-master-instance "${EXISTING_MASTER_ZONE}" "${EXISTING_MASTER_NAME}" "${existing_master_replicas}"

  # Add new replica to the load balancer.
  gcloud compute target-pools add-instances "${MASTER_NAME}" \
    --project "${PROJECT}" \
    --zone "${ZONE}" \
    --instances "${REPLICA_NAME}"
}

# Detaches old and ataches new external IP to a VM.
#
# Arguments:
#   $1 - VM name
#   $2 - VM zone
#   $3 - external static IP; if empty will use an ephemeral IP address.
function attach-external-ip() {
  local NAME=${1}
  local ZONE=${2}
  local IP_ADDR=${3:-}
  local ACCESS_CONFIG_NAME=$(gcloud compute instances describe "${NAME}" \
    --project "${PROJECT}" --zone "${ZONE}" \
    --format="value(networkInterfaces[0].accessConfigs[0].name)")
  gcloud compute instances delete-access-config "${NAME}" \
    --project "${PROJECT}" --zone "${ZONE}" \
    --access-config-name "${ACCESS_CONFIG_NAME}"
  if [[ -z ${IP_ADDR} ]]; then
    gcloud compute instances add-access-config "${NAME}" \
      --project "${PROJECT}" --zone "${ZONE}" \
      --access-config-name "${ACCESS_CONFIG_NAME}"
  else
    gcloud compute instances add-access-config "${NAME}" \
      --project "${PROJECT}" --zone "${ZONE}" \
      --access-config-name "${ACCESS_CONFIG_NAME}" \
      --address "${IP_ADDR}"
  fi
}

# Creates load balancer in front of apiserver if it doesn't exists already. Assumes there's only one
# existing master replica.
#
# Assumes:
#   PROJECT
#   MASTER_NAME
#   ZONE
#   REGION
function create-loadbalancer() {
  detect-master

  # Step 0: Return early if LB is already configured.
  if gcloud compute forwarding-rules describe ${MASTER_NAME} \
    --project "${PROJECT}" --region ${REGION} > /dev/null 2>&1; then
    echo "Load balancer already exists"
    return
  fi

  local EXISTING_MASTER_NAME="$(get-all-replica-names)"
  local EXISTING_MASTER_ZONE=$(gcloud compute instances list "${EXISTING_MASTER_NAME}" \
    --project "${PROJECT}" --format="value(zone)")

  echo "Creating load balancer in front of an already existing master in ${EXISTING_MASTER_ZONE}"

  # Step 1: Detach master IP address and attach ephemeral address to the existing master
  attach-external-ip "${EXISTING_MASTER_NAME}" "${EXISTING_MASTER_ZONE}"

  # Step 2: Create target pool.
  gcloud compute target-pools create "${MASTER_NAME}" --project "${PROJECT}" --region "${REGION}"
  # TODO: We should also add master instances with suffixes
  gcloud compute target-pools add-instances "${MASTER_NAME}" --instances "${EXISTING_MASTER_NAME}" --project "${PROJECT}" --zone "${EXISTING_MASTER_ZONE}"

  # Step 3: Create forwarding rule.
  # TODO: This step can take up to 20 min. We need to speed this up...
  gcloud compute forwarding-rules create ${MASTER_NAME} \
    --project "${PROJECT}" --region ${REGION} \
    --target-pool ${MASTER_NAME} --address=${KUBE_MASTER_IP} --ports=443

  echo -n "Waiting for the load balancer configuration to propagate..."
  local counter=0
  until $(curl -k -m1 https://${KUBE_MASTER_IP} &> /dev/null); do
    counter=$((counter+1))
    echo -n .
    if [[ ${counter} -ge 1800 ]]; then
      echo -e "${color_red}TIMEOUT${color_norm}" >&2
      echo -e "${color_red}Load balancer failed to initialize within ${counter} seconds.${color_norm}" >&2
      exit 2
    fi
  done
  echo "DONE"
}

function create-nodes-firewall() {
  # Create a single firewall rule for all minions.
  create-firewall-rule "${NODE_TAG}-all" "${CLUSTER_IP_RANGE}" "${NODE_TAG}" &

  # Report logging choice (if any).
  if [[ "${ENABLE_NODE_LOGGING-}" == "true" ]]; then
    echo "+++ Logging using Fluentd to ${LOGGING_DESTINATION:-unknown}"
  fi

  # Wait for last batch of jobs
  kube::util::wait-for-jobs || {
    echo -e "${color_red}Some commands failed.${color_norm}" >&2
  }
}

function get-scope-flags() {
  local scope_flags=
  if [[ -n "${NODE_SCOPES}" ]]; then
    scope_flags="--scopes ${NODE_SCOPES}"
  else
    scope_flags="--no-scopes"
  fi
  echo "${scope_flags}"
}

function create-nodes-template() {
  echo "Creating nodes."

  local scope_flags=$(get-scope-flags)

  write-node-env

  local template_name="${NODE_INSTANCE_PREFIX}-template"
  create-node-instance-template $template_name
}

# Assumes:
# - MAX_INSTANCES_PER_MIG
# - NUM_NODES
# exports:
# - NUM_MIGS
function set_num_migs() {
  local defaulted_max_instances_per_mig=${MAX_INSTANCES_PER_MIG:-1000}

  if [[ ${defaulted_max_instances_per_mig} -le "0" ]]; then
    echo "MAX_INSTANCES_PER_MIG cannot be negative. Assuming default 1000"
    defaulted_max_instances_per_mig=1000
  fi
  export NUM_MIGS=$(((${NUM_NODES} + ${defaulted_max_instances_per_mig} - 1) / ${defaulted_max_instances_per_mig}))
}

# Assumes:
# - NUM_MIGS
# - NODE_INSTANCE_PREFIX
# - NUM_NODES
# - PROJECT
# - ZONE
function create-nodes() {
  local template_name="${NODE_INSTANCE_PREFIX}-template"

  if [[ -z "${HEAPSTER_MACHINE_TYPE:-}" ]]; then
    local -r nodes="${NUM_NODES}"
  else
    local -r nodes=$(( NUM_NODES - 1 ))
  fi

  local instances_left=${nodes}

  #TODO: parallelize this loop to speed up the process
  for ((i=1; i<=${NUM_MIGS}; i++)); do
    local group_name="${NODE_INSTANCE_PREFIX}-group-$i"
    if [[ $i == ${NUM_MIGS} ]]; then
      # TODO: We don't add a suffix for the last group to keep backward compatibility when there's only one MIG.
      # We should change it at some point, but note #18545 when changing this.
      group_name="${NODE_INSTANCE_PREFIX}-group"
    fi
    # Spread the remaining number of nodes evenly
    this_mig_size=$((${instances_left} / (${NUM_MIGS}-${i}+1)))
    instances_left=$((instances_left-${this_mig_size}))

    gcloud compute instance-groups managed \
        create "${group_name}" \
        --project "${PROJECT}" \
        --zone "${ZONE}" \
        --base-instance-name "${group_name}" \
        --size "${this_mig_size}" \
        --template "$template_name" || true;
    gcloud compute instance-groups managed wait-until-stable \
        "${group_name}" \
        --zone "${ZONE}" \
        --project "${PROJECT}" || true;
  done

  if [[ -n "${HEAPSTER_MACHINE_TYPE:-}" ]]; then
    echo "Creating a special node for heapster with machine-type ${HEAPSTER_MACHINE_TYPE}"
    create-heapster-node
  fi
}

# Assumes:
# - NODE_INSTANCE_PREFIX
# - PROJECT
# - NETWORK_PROJECT
# - REGION
# - ZONE
# - HEAPSTER_MACHINE_TYPE
# - NODE_DISK_TYPE
# - NODE_DISK_SIZE
# - NODE_IMAGE_PROJECT
# - NODE_IMAGE
# - NODE_SERVICE_ACCOUNT
# - NODE_TAG
# - NETWORK
# - ENABLE_IP_ALIASES
# - SUBNETWORK
# - IP_ALIAS_SIZE
function create-heapster-node() {
  local gcloud="gcloud"

  if [[ "${ENABLE_IP_ALIASES:-}" == 'true' ]]; then
    gcloud="gcloud beta"
  fi

  local network=$(make-gcloud-network-argument \
      "${NETWORK_PROJECT}" \
      "${REGION}" \
      "${NETWORK}" \
      "${SUBNETWORK:-}" \
      "" \
      "${ENABLE_IP_ALIASES:-}" \
      "${IP_ALIAS_SIZE:-}")

  ${gcloud} compute instances \
      create "${NODE_INSTANCE_PREFIX}-heapster" \
      --project "${PROJECT}" \
      --zone "${ZONE}" \
      --machine-type="${HEAPSTER_MACHINE_TYPE}" \
      --boot-disk-type "${NODE_DISK_TYPE}" \
      --boot-disk-size "${NODE_DISK_SIZE}" \
      --image-project="${NODE_IMAGE_PROJECT}" \
      --image "${NODE_IMAGE}" \
      --service-account "${NODE_SERVICE_ACCOUNT}" \
      --tags "${NODE_TAG}" \
      ${network} \
      $(get-scope-flags) \
      --metadata-from-file "$(get-node-instance-metadata)"
}

# Assumes:
# - NUM_MIGS
# - NODE_INSTANCE_PREFIX
# - PROJECT
# - ZONE
# - AUTOSCALER_MAX_NODES
# - AUTOSCALER_MIN_NODES
# Exports
# - AUTOSCALER_MIG_CONFIG
function create-cluster-autoscaler-mig-config() {

  # Each MIG must have at least one node, so the min number of nodes
  # must be greater or equal to the number of migs.
  if [[ ${AUTOSCALER_MIN_NODES} -lt 0 ]]; then
    echo "AUTOSCALER_MIN_NODES must be greater or equal 0"
    exit 2
  fi

  # Each MIG must have at least one node, so the min number of nodes
  # must be greater or equal to the number of migs.
  if [[ ${AUTOSCALER_MAX_NODES} -lt ${NUM_MIGS} ]]; then
    echo "AUTOSCALER_MAX_NODES must be greater or equal ${NUM_MIGS}"
    exit 2
  fi

  # The code assumes that the migs were created with create-nodes
  # function which tries to evenly spread nodes across the migs.
  AUTOSCALER_MIG_CONFIG=""

  local left_min=${AUTOSCALER_MIN_NODES}
  local left_max=${AUTOSCALER_MAX_NODES}

  for ((i=1; i<=${NUM_MIGS}; i++)); do
    local group_name="${NODE_INSTANCE_PREFIX}-group-$i"
    if [[ $i == ${NUM_MIGS} ]]; then
      # TODO: We don't add a suffix for the last group to keep backward compatibility when there's only one MIG.
      # We should change it at some point, but note #18545 when changing this.
      group_name="${NODE_INSTANCE_PREFIX}-group"
    fi

    this_mig_min=$((${left_min}/(${NUM_MIGS}-${i}+1)))
    this_mig_max=$((${left_max}/(${NUM_MIGS}-${i}+1)))
    left_min=$((left_min-$this_mig_min))
    left_max=$((left_max-$this_mig_max))

    local mig_url="https://www.googleapis.com/compute/v1/projects/${PROJECT}/zones/${ZONE}/instanceGroups/${group_name}"
    AUTOSCALER_MIG_CONFIG="${AUTOSCALER_MIG_CONFIG} --nodes=${this_mig_min}:${this_mig_max}:${mig_url}"
  done

  AUTOSCALER_MIG_CONFIG="${AUTOSCALER_MIG_CONFIG} --scale-down-enabled=${AUTOSCALER_ENABLE_SCALE_DOWN}"
}

# Assumes:
# - NUM_MIGS
# - NODE_INSTANCE_PREFIX
# - PROJECT
# - ZONE
# - ENABLE_CLUSTER_AUTOSCALER
# - AUTOSCALER_MAX_NODES
# - AUTOSCALER_MIN_NODES
function create-autoscaler-config() {
  # Create autoscaler for nodes configuration if requested
  if [[ "${ENABLE_CLUSTER_AUTOSCALER}" == "true" ]]; then
    create-cluster-autoscaler-mig-config
    echo "Using autoscaler config: ${AUTOSCALER_MIG_CONFIG} ${AUTOSCALER_EXPANDER_CONFIG}"
  fi
}

function check-cluster() {
  detect-node-names
  detect-master

  echo "Waiting up to ${KUBE_CLUSTER_INITIALIZATION_TIMEOUT} seconds for cluster initialization."
  echo
  echo "  This will continually check to see if the API for kubernetes is reachable."
  echo "  This may time out if there was some uncaught error during start up."
  echo

  # curl in mavericks is borked.
  secure=""
  if which sw_vers >& /dev/null; then
    if [[ $(sw_vers | grep ProductVersion | awk '{print $2}') = "10.9."* ]]; then
      secure="--insecure"
    fi
  fi

  local start_time=$(date +%s)
  local curl_out=$(mktemp)
  kube::util::trap_add "rm -f ${curl_out}" EXIT
  until curl --cacert "${CERT_DIR}/pki/ca.crt" \
          -H "Authorization: Bearer ${KUBE_BEARER_TOKEN}" \
          ${secure} \
          --max-time 5 --fail \
          "https://${KUBE_MASTER_IP}/api/v1/pods?limit=100" > "${curl_out}" 2>&1; do
      local elapsed=$(($(date +%s) - ${start_time}))
      if [[ ${elapsed} -gt ${KUBE_CLUSTER_INITIALIZATION_TIMEOUT} ]]; then
          echo -e "${color_red}Cluster failed to initialize within ${KUBE_CLUSTER_INITIALIZATION_TIMEOUT} seconds.${color_norm}" >&2
          echo "Last output from querying API server follows:" >&2
          echo "-----------------------------------------------------" >&2
          cat "${curl_out}" >&2
          echo "-----------------------------------------------------" >&2
          exit 2
      fi
      printf "."
      sleep 2
  done

  echo "Kubernetes cluster created."

  export KUBE_CERT="${CERT_DIR}/pki/issued/kubecfg.crt"
  export KUBE_KEY="${CERT_DIR}/pki/private/kubecfg.key"
  export CA_CERT="${CERT_DIR}/pki/ca.crt"
  export CONTEXT="${PROJECT}_${INSTANCE_PREFIX}"
  (
   umask 077

   # Update the user's kubeconfig to include credentials for this apiserver.
   create-kubeconfig
  )

  # ensures KUBECONFIG is set
  get-kubeconfig-basicauth
  echo
  echo -e "${color_green}Kubernetes cluster is running.  The master is running at:"
  echo
  echo -e "${color_yellow}  https://${KUBE_MASTER_IP}"
  echo
  echo -e "${color_green}The user name and password to use is located in ${KUBECONFIG}.${color_norm}"
  echo

}

# Removes master replica from etcd cluster.
#
# Assumed vars:
#   REPLICA_NAME
#   PROJECT
#   EXISTING_MASTER_NAME
#   EXISTING_MASTER_ZONE
#
# $1: etcd client port
# returns the result of ssh command which removes replica
function remove-replica-from-etcd() {
  local -r port="${1}"
  [[ -n "${EXISTING_MASTER_NAME}" ]] || return
  gcloud compute ssh "${EXISTING_MASTER_NAME}" \
    --project "${PROJECT}" \
    --zone "${EXISTING_MASTER_ZONE}" \
    --command \
    "curl -s localhost:${port}/v2/members/\$(curl -s localhost:${port}/v2/members -XGET | sed 's/{\\\"id/\n/g' | grep ${REPLICA_NAME}\\\" | cut -f 3 -d \\\") -XDELETE -L 2>/dev/null"
  local -r res=$?
  echo "Removing etcd replica, name: ${REPLICA_NAME}, port: ${port}, result: ${res}"
  return "${res}"
}

# Delete a kubernetes cluster. This is called from test-teardown.
#
# Assumed vars:
#   MASTER_NAME
#   NODE_INSTANCE_PREFIX
#   ZONE
# This function tears down cluster resources 10 at a time to avoid issuing too many
# API calls and exceeding API quota. It is important to bring down the instances before bringing
# down the firewall rules and routes.
function kube-down() {
  local -r batch=200

  detect-project
  detect-node-names # For INSTANCE_GROUPS

  echo "Bringing down cluster"
  set +e  # Do not stop on error

  if [[ "${KUBE_DELETE_NODES:-}" != "false" ]]; then
    # Get the name of the managed instance group template before we delete the
    # managed instance group. (The name of the managed instance group template may
    # change during a cluster upgrade.)
    local templates=$(get-template "${PROJECT}")

    for group in ${INSTANCE_GROUPS[@]:-}; do
      if gcloud compute instance-groups managed describe "${group}" --project "${PROJECT}" --zone "${ZONE}" &>/dev/null; then
        gcloud compute instance-groups managed delete \
          --project "${PROJECT}" \
          --quiet \
          --zone "${ZONE}" \
          "${group}" &
      fi
    done

    # Wait for last batch of jobs
    kube::util::wait-for-jobs || {
      echo -e "Failed to delete instance group(s)." >&2
    }

    for template in ${templates[@]:-}; do
      if gcloud compute instance-templates describe --project "${PROJECT}" "${template}" &>/dev/null; then
        gcloud compute instance-templates delete \
          --project "${PROJECT}" \
          --quiet \
          "${template}"
      fi
    done

    # Delete the special heapster node (if it exists).
    if [[ -n "${HEAPSTER_MACHINE_TYPE:-}" ]]; then
      local -r heapster_machine_name="${NODE_INSTANCE_PREFIX}-heapster"
      if gcloud compute instances describe "${heapster_machine_name}" --zone "${ZONE}" --project "${PROJECT}" &>/dev/null; then
        # Now we can safely delete the VM.
        gcloud compute instances delete \
          --project "${PROJECT}" \
          --quiet \
          --delete-disks all \
          --zone "${ZONE}" \
          "${heapster_machine_name}"
      fi
    fi
  fi

  local -r REPLICA_NAME="${KUBE_REPLICA_NAME:-$(get-replica-name)}"

  set-existing-master

  # Un-register the master replica from etcd and events etcd.
  remove-replica-from-etcd 2379
  remove-replica-from-etcd 4002

  # Delete the master replica (if it exists).
  if gcloud compute instances describe "${REPLICA_NAME}" --zone "${ZONE}" --project "${PROJECT}" &>/dev/null; then
    # If there is a load balancer in front of apiservers we need to first update its configuration.
    if gcloud compute target-pools describe "${MASTER_NAME}" --region "${REGION}" --project "${PROJECT}" &>/dev/null; then
      gcloud compute target-pools remove-instances "${MASTER_NAME}" \
        --project "${PROJECT}" \
        --zone "${ZONE}" \
        --instances "${REPLICA_NAME}"
    fi
    # Now we can safely delete the VM.
    gcloud compute instances delete \
      --project "${PROJECT}" \
      --quiet \
      --delete-disks all \
      --zone "${ZONE}" \
      "${REPLICA_NAME}"
  fi

  # Delete the master replica pd (possibly leaked by kube-up if master create failed).
  # TODO(jszczepkowski): remove also possibly leaked replicas' pds
  local -r replica_pd="${REPLICA_NAME:-${MASTER_NAME}}-pd"
  if gcloud compute disks describe "${replica_pd}" --zone "${ZONE}" --project "${PROJECT}" &>/dev/null; then
    gcloud compute disks delete \
      --project "${PROJECT}" \
      --quiet \
      --zone "${ZONE}" \
      "${replica_pd}"
  fi

  # Check if this are any remaining master replicas.
  local REMAINING_MASTER_COUNT=$(gcloud compute instances list \
    --project "${PROJECT}" \
    --filter="name ~ '$(get-replica-name-regexp)'" \
    --format "value(zone)" | wc -l)

  # In the replicated scenario, if there's only a single master left, we should also delete load balancer in front of it.
  if [[ "${REMAINING_MASTER_COUNT}" -eq 1 ]]; then
    if gcloud compute forwarding-rules describe "${MASTER_NAME}" --region "${REGION}" --project "${PROJECT}" &>/dev/null; then
      detect-master
      local REMAINING_REPLICA_NAME="$(get-all-replica-names)"
      local REMAINING_REPLICA_ZONE=$(gcloud compute instances list "${REMAINING_REPLICA_NAME}" \
        --project "${PROJECT}" --format="value(zone)")
      gcloud compute forwarding-rules delete \
        --project "${PROJECT}" \
        --region "${REGION}" \
        --quiet \
        "${MASTER_NAME}"
      attach-external-ip "${REMAINING_REPLICA_NAME}" "${REMAINING_REPLICA_ZONE}" "${KUBE_MASTER_IP}"
      gcloud compute target-pools delete \
        --project "${PROJECT}" \
        --region "${REGION}" \
        --quiet \
        "${MASTER_NAME}"
    fi
  fi

  # If there are no more remaining master replicas, we should delete all remaining network resources.
  if [[ "${REMAINING_MASTER_COUNT}" -eq 0 ]]; then
    # Delete firewall rule for the master, etcd servers, and nodes.
    delete-firewall-rules "${MASTER_NAME}-https" "${MASTER_NAME}-etcd" "${NODE_TAG}-all"
    # Delete the master's reserved IP
    if gcloud compute addresses describe "${MASTER_NAME}-ip" --region "${REGION}" --project "${PROJECT}" &>/dev/null; then
      gcloud compute addresses delete \
        --project "${PROJECT}" \
        --region "${REGION}" \
        --quiet \
        "${MASTER_NAME}-ip"
    fi
  fi

  if [[ "${KUBE_DELETE_NODES:-}" != "false" ]]; then
    # Find out what minions are running.
    local -a minions
    minions=( $(gcloud compute instances list \
                  --project "${PROJECT}" \
                  --filter="name ~ '${NODE_INSTANCE_PREFIX}-.+' AND zone:(${ZONE})" \
                  --format='value(name)') )
    # If any minions are running, delete them in batches.
    while (( "${#minions[@]}" > 0 )); do
      echo Deleting nodes "${minions[*]::${batch}}"
      gcloud compute instances delete \
        --project "${PROJECT}" \
        --quiet \
        --delete-disks boot \
        --zone "${ZONE}" \
        "${minions[@]::${batch}}"
      minions=( "${minions[@]:${batch}}" )
    done
  fi

  # If there are no more remaining master replicas: delete routes, pd for influxdb and update kubeconfig
  if [[ "${REMAINING_MASTER_COUNT}" -eq 0 ]]; then
    # Delete routes.
    local -a routes
    # Clean up all routes w/ names like "<cluster-name>-<node-GUID>"
    # e.g. "kubernetes-12345678-90ab-cdef-1234-567890abcdef". The name is
    # determined by the node controller on the master.
    # Note that this is currently a noop, as synchronously deleting the node MIG
    # first allows the master to cleanup routes itself.
    local TRUNCATED_PREFIX="${INSTANCE_PREFIX:0:26}"
    routes=( $(gcloud compute routes list --project "${NETWORK_PROJECT}" \
      --filter="name ~ '${TRUNCATED_PREFIX}-.{8}-.{4}-.{4}-.{4}-.{12}'" \
      --format='value(name)') )
    while (( "${#routes[@]}" > 0 )); do
      echo Deleting routes "${routes[*]::${batch}}"
      gcloud compute routes delete \
        --project "${NETWORK_PROJECT}" \
        --quiet \
        "${routes[@]::${batch}}"
      routes=( "${routes[@]:${batch}}" )
    done

    # Delete persistent disk for influx-db.
    if gcloud compute disks describe "${INSTANCE_PREFIX}"-influxdb-pd --zone "${ZONE}" --project "${PROJECT}" &>/dev/null; then
      gcloud compute disks delete \
        --project "${PROJECT}" \
        --quiet \
        --zone "${ZONE}" \
        "${INSTANCE_PREFIX}"-influxdb-pd
    fi

    # Delete all remaining firewall rules and network.
    delete-firewall-rules \
      "${CLUSTER_NAME}-default-internal-master" \
      "${CLUSTER_NAME}-default-internal-node" \
      "${NETWORK}-default-ssh" \
      "${NETWORK}-default-internal"  # Pre-1.5 clusters

    if [[ "${KUBE_DELETE_NETWORK}" == "true" ]]; then
      # Delete all remaining firewall rules in the network.
      delete-all-firewall-rules || true
      delete-subnetworks || true
      delete-network || true  # might fail if there are leaked resources that reference the network
    fi

    # If there are no more remaining master replicas, we should update kubeconfig.
    export CONTEXT="${PROJECT}_${INSTANCE_PREFIX}"
    clear-kubeconfig
  else
  # If some master replicas remain: cluster has been changed, we need to re-validate it.
    echo "... calling validate-cluster" >&2
    # Override errexit
    (validate-cluster) && validate_result="$?" || validate_result="$?"

    # We have two different failure modes from validate cluster:
    # - 1: fatal error - cluster won't be working correctly
    # - 2: weak error - something went wrong, but cluster probably will be working correctly
    # We just print an error message in case 2).
    if [[ "${validate_result}" -eq 1 ]]; then
      exit 1
    elif [[ "${validate_result}" -eq 2 ]]; then
      echo "...ignoring non-fatal errors in validate-cluster" >&2
    fi
  fi
  set -e
}

# Prints name of one of the master replicas in the current zone. It will be either
# just MASTER_NAME or MASTER_NAME with a suffix for a replica (see get-replica-name-regexp).
#
# Assumed vars:
#   PROJECT
#   ZONE
#   MASTER_NAME
#
# NOTE: Must be in sync with get-replica-name-regexp and set-replica-name.
function get-replica-name() {
  echo $(gcloud compute instances list \
    --project "${PROJECT}" \
    --filter="name ~ '$(get-replica-name-regexp)' AND zone:(${ZONE})" \
    --format "value(name)" | head -n1)
}

# Prints comma-separated names of all of the master replicas in all zones.
#
# Assumed vars:
#   PROJECT
#   MASTER_NAME
#
# NOTE: Must be in sync with get-replica-name-regexp and set-replica-name.
function get-all-replica-names() {
  echo $(gcloud compute instances list \
    --project "${PROJECT}" \
    --filter="name ~ '$(get-replica-name-regexp)'" \
    --format "value(name)" | tr "\n" "," | sed 's/,$//')
}

# Prints the number of all of the master replicas in all zones.
#
# Assumed vars:
#   MASTER_NAME
function get-master-replicas-count() {
  detect-project
  local num_masters=$(gcloud compute instances list \
    --project "${PROJECT}" \
    --filter="name ~ '$(get-replica-name-regexp)'" \
    --format "value(zone)" | wc -l)
  echo -n "${num_masters}"
}

# Prints regexp for full master machine name. In a cluster with replicated master,
# VM names may either be MASTER_NAME or MASTER_NAME with a suffix for a replica.
function get-replica-name-regexp() {
  echo "^${MASTER_NAME}(-...)?"
}

# Sets REPLICA_NAME to a unique name for a master replica that will match
# expected regexp (see get-replica-name-regexp).
#
# Assumed vars:
#   PROJECT
#   ZONE
#   MASTER_NAME
#
# Sets:
#   REPLICA_NAME
function set-replica-name() {
  local instances=$(gcloud compute instances list \
    --project "${PROJECT}" \
    --filter="name ~ '$(get-replica-name-regexp)'" \
    --format "value(name)")

  suffix=""
  while echo "${instances}" | grep "${suffix}" &>/dev/null; do
    suffix="$(date | md5sum | head -c3)"
  done
  REPLICA_NAME="${MASTER_NAME}-${suffix}"
}

# Gets the instance template for given NODE_INSTANCE_PREFIX. It echos the template name so that the function
# output can be used.
# Assumed vars:
#   NODE_INSTANCE_PREFIX
#
# $1: project
function get-template() {
  gcloud compute instance-templates list \
    --filter="name ~ '${NODE_INSTANCE_PREFIX}-template(-(${KUBE_RELEASE_VERSION_DASHED_REGEX}|${KUBE_CI_VERSION_DASHED_REGEX}))?'" \
    --project="${1}" --format='value(name)'
}

# Checks if there are any present resources related kubernetes cluster.
#
# Assumed vars:
#   MASTER_NAME
#   NODE_INSTANCE_PREFIX
#   ZONE
#   REGION
# Vars set:
#   KUBE_RESOURCE_FOUND
function check-resources() {
  detect-project
  detect-node-names

  echo "Looking for already existing resources"
  KUBE_RESOURCE_FOUND=""

  if [[ -n "${INSTANCE_GROUPS[@]:-}" ]]; then
    KUBE_RESOURCE_FOUND="Managed instance groups ${INSTANCE_GROUPS[@]}"
    return 1
  fi

  if gcloud compute instance-templates describe --project "${PROJECT}" "${NODE_INSTANCE_PREFIX}-template" &>/dev/null; then
    KUBE_RESOURCE_FOUND="Instance template ${NODE_INSTANCE_PREFIX}-template"
    return 1
  fi

  if gcloud compute instances describe --project "${PROJECT}" "${MASTER_NAME}" --zone "${ZONE}" &>/dev/null; then
    KUBE_RESOURCE_FOUND="Kubernetes master ${MASTER_NAME}"
    return 1
  fi

  if gcloud compute disks describe --project "${PROJECT}" "${MASTER_NAME}"-pd --zone "${ZONE}" &>/dev/null; then
    KUBE_RESOURCE_FOUND="Persistent disk ${MASTER_NAME}-pd"
    return 1
  fi

  # Find out what minions are running.
  local -a minions
  minions=( $(gcloud compute instances list \
                --project "${PROJECT}" \
                --filter="name ~ '${NODE_INSTANCE_PREFIX}-.+' AND zone:(${ZONE})" \
                --format='value(name)') )
  if (( "${#minions[@]}" > 0 )); then
    KUBE_RESOURCE_FOUND="${#minions[@]} matching matching ${NODE_INSTANCE_PREFIX}-.+"
    return 1
  fi

  if gcloud compute firewall-rules describe --project "${NETWORK_PROJECT}" "${MASTER_NAME}-https" &>/dev/null; then
    KUBE_RESOURCE_FOUND="Firewall rules for ${MASTER_NAME}-https"
    return 1
  fi

  if gcloud compute firewall-rules describe --project "${NETWORK_PROJECT}" "${NODE_TAG}-all" &>/dev/null; then
    KUBE_RESOURCE_FOUND="Firewall rules for ${MASTER_NAME}-all"
    return 1
  fi

  local -a routes
  routes=( $(gcloud compute routes list --project "${NETWORK_PROJECT}" \
    --filter="name ~ '${INSTANCE_PREFIX}-minion-.{4}'" --format='value(name)') )
  if (( "${#routes[@]}" > 0 )); then
    KUBE_RESOURCE_FOUND="${#routes[@]} routes matching ${INSTANCE_PREFIX}-minion-.{4}"
    return 1
  fi

  if gcloud compute addresses describe --project "${PROJECT}" "${MASTER_NAME}-ip" --region "${REGION}" &>/dev/null; then
    KUBE_RESOURCE_FOUND="Master's reserved IP"
    return 1
  fi

  # No resources found.
  return 0
}

# Prepare to push new binaries to kubernetes cluster
#  $1 - whether prepare push to node
function prepare-push() {
  local node="${1-}"
  #TODO(dawnchen): figure out how to upgrade a Container Linux node
  if [[ "${node}" == "true" && "${NODE_OS_DISTRIBUTION}" != "debian" ]]; then
    echo "Updating nodes in a kubernetes cluster with ${NODE_OS_DISTRIBUTION} is not supported yet." >&2
    exit 1
  fi
  if [[ "${node}" != "true" && "${MASTER_OS_DISTRIBUTION}" != "debian" ]]; then
    echo "Updating the master in a kubernetes cluster with ${MASTER_OS_DISTRIBUTION} is not supported yet." >&2
    exit 1
  fi

  OUTPUT=${KUBE_ROOT}/_output/logs
  mkdir -p ${OUTPUT}

  kube::util::ensure-temp-dir
  detect-project
  detect-master
  detect-node-names
  get-kubeconfig-basicauth
  get-kubeconfig-bearertoken

  # Make sure we have the tar files staged on Google Storage
  tars_from_version

  # Prepare node env vars and update MIG template
  if [[ "${node}" == "true" ]]; then
    write-node-env

    local scope_flags=$(get-scope-flags)

    # Ugly hack: Since it is not possible to delete instance-template that is currently
    # being used, create a temp one, then delete the old one and recreate it once again.
    local tmp_template_name="${NODE_INSTANCE_PREFIX}-template-tmp"
    create-node-instance-template $tmp_template_name

    local template_name="${NODE_INSTANCE_PREFIX}-template"
    for group in ${INSTANCE_GROUPS[@]:-}; do
      gcloud compute instance-groups managed \
        set-instance-template "${group}" \
        --template "$tmp_template_name" \
        --zone "${ZONE}" \
        --project "${PROJECT}" || true;
    done

    gcloud compute instance-templates delete \
      --project "${PROJECT}" \
      --quiet \
      "$template_name" || true

    create-node-instance-template "$template_name"

    for group in ${INSTANCE_GROUPS[@]:-}; do
      gcloud compute instance-groups managed \
        set-instance-template "${group}" \
        --template "$template_name" \
        --zone "${ZONE}" \
        --project "${PROJECT}" || true;
    done

    gcloud compute instance-templates delete \
      --project "${PROJECT}" \
      --quiet \
      "$tmp_template_name" || true
  fi
}

# -----------------------------------------------------------------------------
# Cluster specific test helpers used from hack/e2e.go

# Execute prior to running tests to build a release if required for env.
#
# Assumed Vars:
#   KUBE_ROOT
function test-build-release() {
  # Make a release
  "${KUBE_ROOT}/build/release.sh"
}

# Execute prior to running tests to initialize required structure. This is
# called from hack/e2e.go only when running -up.
#
# Assumed vars:
#   Variables from config.sh
function test-setup() {
  # Detect the project into $PROJECT if it isn't set
  detect-project

  if [[ ${MULTIZONE:-} == "true" && -n ${E2E_ZONES:-} ]]; then
    for KUBE_GCE_ZONE in ${E2E_ZONES}; do
      KUBE_GCE_ZONE="${KUBE_GCE_ZONE}" KUBE_USE_EXISTING_MASTER="${KUBE_USE_EXISTING_MASTER:-}" "${KUBE_ROOT}/cluster/kube-up.sh"
      KUBE_USE_EXISTING_MASTER="true" # For subsequent zones we use the existing master
    done
  else
    "${KUBE_ROOT}/cluster/kube-up.sh"
  fi

  # Open up port 80 & 8080 so common containers on minions can be reached
  # TODO(roberthbailey): Remove this once we are no longer relying on hostPorts.
  local start=`date +%s`
  gcloud compute firewall-rules create \
    --project "${NETWORK_PROJECT}" \
    --target-tags "${NODE_TAG}" \
    --allow tcp:80,tcp:8080 \
    --network "${NETWORK}" \
    "${NODE_TAG}-${INSTANCE_PREFIX}-http-alt" 2> /dev/null || true
  # As there is no simple way to wait longer for this operation we need to manually
  # wait some additional time (20 minutes altogether).
  while ! gcloud compute firewall-rules describe --project "${NETWORK_PROJECT}" "${NODE_TAG}-${INSTANCE_PREFIX}-http-alt" 2> /dev/null; do
    if [[ $(($start + 1200)) -lt `date +%s` ]]; then
      echo -e "${color_red}Failed to create firewall ${NODE_TAG}-${INSTANCE_PREFIX}-http-alt in ${NETWORK_PROJECT}" >&2
      exit 1
    fi
    sleep 5
  done

  # Open up the NodePort range
  # TODO(justinsb): Move to main setup, if we decide whether we want to do this by default.
  start=`date +%s`
  gcloud compute firewall-rules create \
    --project "${NETWORK_PROJECT}" \
    --target-tags "${NODE_TAG}" \
    --allow tcp:30000-32767,udp:30000-32767 \
    --network "${NETWORK}" \
    "${NODE_TAG}-${INSTANCE_PREFIX}-nodeports" 2> /dev/null || true
  # As there is no simple way to wait longer for this operation we need to manually
  # wait some additional time (20 minutes altogether).
  while ! gcloud compute firewall-rules describe --project "${NETWORK_PROJECT}" "${NODE_TAG}-${INSTANCE_PREFIX}-nodeports" 2> /dev/null; do
    if [[ $(($start + 1200)) -lt `date +%s` ]]; then
      echo -e "${color_red}Failed to create firewall ${NODE_TAG}-${INSTANCE_PREFIX}-nodeports in ${PROJECT}" >&2
      exit 1
    fi
    sleep 5
  done
}

# Execute after running tests to perform any required clean-up. This is called
# from hack/e2e.go
function test-teardown() {
  detect-project
  echo "Shutting down test cluster in background."
  delete-firewall-rules \
    "${NODE_TAG}-${INSTANCE_PREFIX}-http-alt" \
    "${NODE_TAG}-${INSTANCE_PREFIX}-nodeports"
  if [[ ${MULTIZONE:-} == "true" && -n ${E2E_ZONES:-} ]]; then
    local zones=( ${E2E_ZONES} )
    # tear them down in reverse order, finally tearing down the master too.
    for ((zone_num=${#zones[@]}-1; zone_num>0; zone_num--)); do
      KUBE_GCE_ZONE="${zones[zone_num]}" KUBE_USE_EXISTING_MASTER="true" "${KUBE_ROOT}/cluster/kube-down.sh"
    done
    KUBE_GCE_ZONE="${zones[0]}" KUBE_USE_EXISTING_MASTER="false" "${KUBE_ROOT}/cluster/kube-down.sh"
  else
    "${KUBE_ROOT}/cluster/kube-down.sh"
  fi
}

# SSH to a node by name ($1) and run a command ($2).
function ssh-to-node() {
  local node="$1"
  local cmd="$2"
  # Loop until we can successfully ssh into the box
  for try in {1..5}; do
    if gcloud compute ssh --ssh-flag="-o LogLevel=quiet" --ssh-flag="-o ConnectTimeout=30" --project "${PROJECT}" --zone="${ZONE}" "${node}" --command "echo test > /dev/null"; then
      break
    fi
    sleep 5
  done
  # Then actually try the command.
  gcloud compute ssh --ssh-flag="-o LogLevel=quiet" --ssh-flag="-o ConnectTimeout=30" --project "${PROJECT}" --zone="${ZONE}" "${node}" --command "${cmd}"
}

# Perform preparations required to run e2e tests
function prepare-e2e() {
  detect-project
}
