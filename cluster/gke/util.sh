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

# A library of helper functions and constant for the local config.

# Uses the config file specified in $KUBE_CONFIG_FILE, or defaults to config-default.sh

KUBE_PROMPT_FOR_UPDATE=${KUBE_PROMPT_FOR_UPDATE:-"n"}
KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/cluster/gke/${KUBE_CONFIG_FILE:-config-default.sh}"
source "${KUBE_ROOT}/cluster/common.sh"
source "${KUBE_ROOT}/cluster/lib/util.sh"

function with-retry() {
  local retry_limit=$1
  local cmd=("${@:2}")

  local retry_count=0
  local rc=0

  until [[ ${retry_count} -ge ${retry_limit} ]]; do
    ((retry_count+=1))
    "${cmd[@]}" && rc=0 || rc=$?
    if [[ ${rc} == 0 ]]; then
      return 0
    fi
    sleep 3
  done

  return ${rc}
}

# Perform preparations required to run e2e tests
#
# Assumed vars:
#   GCLOUD
function prepare-e2e() {
  echo "... in gke:prepare-e2e()" >&2

  # Ensure GCLOUD is set to some gcloud binary.
  if [[ -z "${GCLOUD:-}" ]]; then
    echo "GCLOUD environment variable is not set. It should be your gcloud binary. " >&2
    echo "A sane default is probably \$ export GCLOUD=gcloud" >&2
    exit 1
  fi
}


# Use the gcloud defaults to find the project.  If it is already set in the
# environment then go with that.
#
# Assumed vars:
#   GCLOUD
# Vars set:
#   PROJECT
function detect-project() {
  echo "... in gke:detect-project()" >&2
  if [[ -z "${PROJECT:-}" ]]; then
    export PROJECT=$("${GCLOUD}" config list project --format 'value(core.project)')
    echo "... Using project: ${PROJECT}" >&2
  fi
  if [[ -z "${PROJECT:-}" ]]; then
    echo "Could not detect Google Cloud Platform project. Set the default project using " >&2
    echo "'gcloud config set project <PROJECT>'" >&2
    exit 1
  fi
}

# Execute prior to running tests to build a release if required for env.
#
# Assumed Vars:
#   KUBE_ROOT
function test-build-release() {
  echo "... in gke:test-build-release()" >&2
  "${KUBE_ROOT}/build/release.sh"
}

# Verify needed binaries exist.
function verify-prereqs() {
  echo "... in gke:verify-prereqs()" >&2
  if ! which gcloud >/dev/null; then
    local resp
    if [[ "${KUBE_PROMPT_FOR_UPDATE}" == "y" ]]; then
      echo "Can't find gcloud in PATH.  Do you wish to install the Google Cloud SDK? [Y/n]"
      read resp
    fi
    if [[ "${resp}" != "n" && "${resp}" != "N" ]]; then
      curl https://sdk.cloud.google.com | bash
    fi
    if ! which gcloud >/dev/null; then
      echo "Can't find gcloud in PATH, please fix and retry. The Google Cloud "
      echo "SDK can be downloaded from https://cloud.google.com/sdk/."
      exit 1
    fi
  fi
  update-or-verify-gcloud
}

# Validate a kubernetes cluster
function validate-cluster {
  # Simply override the NUM_NODES variable if we've spread nodes across multiple
  # zones before calling into the generic validate-cluster logic.
  local EXPECTED_NUM_NODES="${NUM_NODES}"
  for zone in $(echo "${ADDITIONAL_ZONES}" | sed "s/,/ /g")
  do
    (( EXPECTED_NUM_NODES += NUM_NODES ))
  done
  NUM_NODES=${EXPECTED_NUM_NODES} bash -c "${KUBE_ROOT}/cluster/validate-cluster.sh"
}

# Instantiate a kubernetes cluster
#
# Assumed vars:
#   GCLOUD
#   CLUSTER_NAME
#   ZONE
#   CLUSTER_API_VERSION (optional)
#   NUM_NODES
#   ADDITIONAL_ZONES (optional)
#   NODE_SCOPES
#   MACHINE_TYPE
#   HEAPSTER_MACHINE_TYPE (optional)
#   CLUSTER_IP_RANGE (optional)
#   GKE_CREATE_FLAGS (optional, space delineated)
function kube-up() {
  echo "... in gke:kube-up()" >&2
  detect-project >&2

  # Make the specified network if we need to.
  if ! "${GCLOUD}" compute networks --project "${PROJECT}" describe "${NETWORK}" &>/dev/null; then
    echo "Creating new network: ${NETWORK}" >&2
    with-retry 3 "${GCLOUD}" compute networks create "${NETWORK}" --project="${PROJECT}" --range "${NETWORK_RANGE}"
  else
    echo "... Using network: ${NETWORK}" >&2
  fi

  # Allow SSH on all nodes in the network. This doesn't actually check whether
  # such a rule exists, only whether we've created this exact rule.
  if ! "${GCLOUD}" compute firewall-rules --project "${PROJECT}" describe "${FIREWALL_SSH}" &>/dev/null; then
    echo "Creating new firewall for SSH: ${FIREWALL_SSH}" >&2
    with-retry 3 "${GCLOUD}" compute firewall-rules create "${FIREWALL_SSH}" \
      --allow="tcp:22" \
      --network="${NETWORK}" \
      --project="${PROJECT}" \
      --source-ranges="0.0.0.0/0"
  else
    echo "... Using firewall-rule: ${FIREWALL_SSH}" >&2
  fi

  local shared_args=(
    "--zone=${ZONE}"
    "--project=${PROJECT}"
    "--scopes=${NODE_SCOPES}"
  )

  if [[ ! -z "${IMAGE_TYPE:-}" ]]; then
    shared_args+=("--image-type=${IMAGE_TYPE}")
  fi

  if [[ -z "${HEAPSTER_MACHINE_TYPE:-}" ]]; then
    local -r nodes="${NUM_NODES}"
  else
    local -r nodes=$(( NUM_NODES - 1 ))
  fi

  local create_args=(
    ${shared_args[@]}
    "--num-nodes=${nodes}"
    "--network=${NETWORK}"
    "--cluster-version=${CLUSTER_API_VERSION}"
    "--machine-type=${MACHINE_TYPE}"
  )

  if [[ ! -z "${ADDITIONAL_ZONES:-}" ]]; then
    create_args+=("--additional-zones=${ADDITIONAL_ZONES}")
  fi

  if [[ ! -z "${CLUSTER_IP_RANGE:-}" ]]; then
    create_args+=("--cluster-ipv4-cidr=${CLUSTER_IP_RANGE}")
  fi

  create_args+=( ${GKE_CREATE_FLAGS:-} )

  # Bring up the cluster.
  "${GCLOUD}" ${CMD_GROUP:-} container clusters create "${CLUSTER_NAME}" "${create_args[@]}"

  create-kubeconfig-for-federation

  if [[ ! -z "${HEAPSTER_MACHINE_TYPE:-}" ]]; then
    "${GCLOUD}" ${CMD_GROUP:-} container node-pools create "heapster-pool" --cluster "${CLUSTER_NAME}" --num-nodes=1 --machine-type="${HEAPSTER_MACHINE_TYPE}" "${shared_args[@]}"
  fi
}

# Execute prior to running tests to initialize required structure. This is
# called from hack/e2e-go only when running -up (it is run after kube-up, so
# the cluster already exists at this point).
#
# Assumed vars:
#   CLUSTER_NAME
#   GCLOUD
#   ZONE
# Vars set:
#   NODE_TAG
function test-setup() {
  echo "... in gke:test-setup()" >&2
  # Detect the project into $PROJECT if it isn't set
  detect-project >&2

  "${KUBE_ROOT}/cluster/kube-up.sh"

  detect-nodes >&2

  # At this point, CLUSTER_NAME should have been used, so its value is final.
  NODE_TAG=$($GCLOUD compute instances describe ${NODE_NAMES[0]} --project="${PROJECT}" --zone="${ZONE}" --format='value(tags.items)' | grep -o "gke-${CLUSTER_NAME}-.\{8\}-node")
  OLD_NODE_TAG="k8s-${CLUSTER_NAME}-node"

  # Open up port 80 & 8080 so common containers on minions can be reached.
  with-retry 3 "${GCLOUD}" compute firewall-rules create \
    "${CLUSTER_NAME}-http-alt" \
    --allow tcp:80,tcp:8080 \
    --project "${PROJECT}" \
    --target-tags "${NODE_TAG},${OLD_NODE_TAG}" \
    --network="${NETWORK}" &

  with-retry 3 "${GCLOUD}" compute firewall-rules create \
    "${CLUSTER_NAME}-nodeports" \
    --allow tcp:30000-32767,udp:30000-32767 \
    --project "${PROJECT}" \
    --target-tags "${NODE_TAG},${OLD_NODE_TAG}" \
    --network="${NETWORK}" &

  # Wait for firewall rules.
  kube::util::wait-for-jobs || {
    echo "... gke:test-setup(): Could not create firewall" >&2
    return 1
  }
}

# Detect the IP for the master. Note that on GKE, we don't know the name of the
# master, so KUBE_MASTER is not set.
#
# Assumed vars:
#   ZONE
#   CLUSTER_NAME
# Vars set:
#   KUBE_MASTER_IP
function detect-master() {
  echo "... in gke:detect-master()" >&2
  detect-project >&2
  KUBE_MASTER_IP=$("${GCLOUD}" ${CMD_GROUP:-} container clusters describe \
    --project="${PROJECT}" --zone="${ZONE}" --format='value(endpoint)' \
    "${CLUSTER_NAME}")
}

# Assumed vars:
#   none
# Vars set:
#   NODE_NAMES
function detect-nodes() {
  echo "... in gke:detect-nodes()" >&2
  detect-node-names
}

# Detect minions created in the minion group
#
# Note that this will only select nodes in the same zone as the
# cluster, meaning that it won't include all nodes in a multi-zone cluster.
#
# Assumed vars:
#   none
# Vars set:
#   NODE_NAMES
function detect-node-names {
  echo "... in gke:detect-node-names()" >&2
  detect-project
  detect-node-instance-groups

  NODE_NAMES=()
  for group in "${NODE_INSTANCE_GROUPS[@]:-}"; do
    NODE_NAMES+=($(gcloud compute instance-groups managed list-instances \
      "${group}" --zone "${ZONE}" \
      --project "${PROJECT}" --format='value(instance)'))
  done
  echo "NODE_NAMES=${NODE_NAMES[*]:-}"
}

# Detect instance group name generated by gke.
#
# Note that the NODE_INSTANCE_GROUPS var will only have instance groups in the
# same zone as the cluster, meaning that it won't include all groups in a
# multi-zone cluster. The ALL_INSTANCE_GROUP_URLS will contain all the
# instance group URLs, which include multi-zone groups.
#
# Assumed vars:
#   GCLOUD
#   PROJECT
#   ZONE
#   CLUSTER_NAME
# Vars set:
#   NODE_INSTANCE_GROUPS
#   ALL_INSTANCE_GROUP_URLS
function detect-node-instance-groups {
  echo "... in gke:detect-node-instance-groups()" >&2
  local urls=$("${GCLOUD}" ${CMD_GROUP:-} container clusters describe \
    --project="${PROJECT}" --zone="${ZONE}" \
    --format='value(instanceGroupUrls)' "${CLUSTER_NAME}")
  urls=(${urls//;/ })
  ALL_INSTANCE_GROUP_URLS=${urls[*]}
  NODE_INSTANCE_GROUPS=()
  for url in "${urls[@]:-}"; do
    local igm_zone=$(expr ${url} : '.*/zones/\([a-z0-9-]*\)/')
    if [[ "${igm_zone}" == "${ZONE}" ]]; then
      NODE_INSTANCE_GROUPS+=("${url##*/}")
    fi
  done
}

# SSH to a node by name ($1) and run a command ($2).
#
# Assumed vars:
#   GCLOUD
#   ZONE
function ssh-to-node() {
  echo "... in gke:ssh-to-node()" >&2
  detect-project >&2

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

# Execute after running tests to perform any required clean-up.  This is called
# from hack/e2e.go. This calls kube-down, so the cluster still exists when this
# is called.
#
# Assumed vars:
#   CLUSTER_NAME
#   GCLOUD
#   KUBE_ROOT
#   ZONE
function test-teardown() {
  echo "... in gke:test-teardown()" >&2

  detect-project >&2

  # Tear down the cluster first.
  "${KUBE_ROOT}/cluster/kube-down.sh" || true

  # Then remove the firewall rules. We do it in this order because the
  # time to delete a firewall is actually dependent on the number of
  # instances, but we can safely delete the cluster before the firewall.
  #
  # NOTE: Keep in sync with names above in test-setup.
  for fw in "${CLUSTER_NAME}-http-alt" "${CLUSTER_NAME}-nodeports" "${FIREWALL_SSH}"; do
    if [[ -n $("${GCLOUD}" compute firewall-rules --project "${PROJECT}" describe "${fw}" --format='value(name)' 2>/dev/null || true) ]]; then
      with-retry 3 "${GCLOUD}" compute firewall-rules delete "${fw}" --project="${PROJECT}" --quiet &
    fi
  done

  # Wait for firewall rule teardown.
  kube::util::wait-for-jobs || true

  # It's unfortunate that the $FIREWALL_SSH rule and network are created in
  # kube-up, but we can only really delete them in test-teardown. So much for
  # symmetry.
  if [[ "${KUBE_DELETE_NETWORK}" == "true" ]]; then
    if [[ -n $("${GCLOUD}" compute networks --project "${PROJECT}" describe "${NETWORK}" --format='value(name)' 2>/dev/null || true) ]]; then
      if ! with-retry 3 "${GCLOUD}" compute networks delete --project "${PROJECT}" --quiet "${NETWORK}"; then
        echo "Failed to delete network '${NETWORK}'. Listing firewall-rules:"
        "${GCLOUD}" compute firewall-rules --project "${PROJECT}" list --filter="network=${NETWORK}"
      fi
    fi
  fi
}

# Actually take down the cluster. This is called from test-teardown.
#
# Assumed vars:
#  GCLOUD
#  ZONE
#  CLUSTER_NAME
function kube-down() {
  echo "... in gke:kube-down()" >&2
  detect-project >&2
  if "${GCLOUD}" ${CMD_GROUP:-} container clusters describe --project="${PROJECT}" --zone="${ZONE}" "${CLUSTER_NAME}" --quiet &>/dev/null; then
    with-retry 3 "${GCLOUD}" ${CMD_GROUP:-} container clusters delete --project="${PROJECT}" \
      --zone="${ZONE}" "${CLUSTER_NAME}" --quiet
  fi
}
