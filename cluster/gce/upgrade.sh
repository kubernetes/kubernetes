#!/usr/bin/env bash

# Copyright 2015 The Kubernetes Authors.
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

# !!!EXPERIMENTAL !!! Upgrade script for GCE. Expect this to get
# rewritten in Go in relatively short order, but it allows us to start
# testing the concepts.

set -o errexit
set -o nounset
set -o pipefail

if [[ "${KUBERNETES_PROVIDER:-gce}" != "gce" ]]; then
  echo "!!! ${1} only works on GCE" >&2
  exit 1
fi

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/hack/lib/util.sh"
source "${KUBE_ROOT}/cluster/kube-util.sh"

function usage() {
  echo "!!! EXPERIMENTAL !!!"
  echo ""
  echo "${0} [-M | -N | -P] [-o] (-l | <version number or publication>)"
  echo "  Upgrades master and nodes by default"
  echo "  -M:  Upgrade master only"
  echo "  -N:  Upgrade nodes only"
  echo "  -P:  Node upgrade prerequisites only (create a new instance template)"
  echo "  -c:  Upgrade NODE_UPGRADE_PARALLELISM nodes in parallel (default=1) within a single instance group. The MIGs themselves are dealt serially."
  echo "  -o:  Use os distro specified in KUBE_NODE_OS_DISTRIBUTION for new nodes. Options include 'debian' or 'gci'"
  echo "  -l:  Use local(dev) binaries. This is only supported for master upgrades."
  echo ""
  echo '  Version number or publication is either a proper version number'
  echo '  (e.g. "v1.0.6", "v1.2.0-alpha.1.881+376438b69c7612") or a version'
  echo '  publication of the form <bucket>/<version> (e.g. "release/stable",'
  echo '  "ci/latest-1").  Some common ones are:'
  echo '    - "release/stable"'
  echo '    - "release/latest"'
  echo '    - "ci/latest"'
  echo '  See the docs on getting builds for more information about version publication.'
  echo ""
  echo "(... Fetching current release versions ...)"
  echo ""

  # NOTE: IF YOU CHANGE THE FOLLOWING LIST, ALSO UPDATE test/e2e/cluster_upgrade.go
  local release_stable
  local release_latest
  local ci_latest

  release_stable=$(gsutil cat gs://kubernetes-release/release/stable.txt)
  release_latest=$(gsutil cat gs://kubernetes-release/release/latest.txt)
  ci_latest=$(gsutil cat gs://kubernetes-release-dev/ci/latest.txt)

  echo "Right now, versions are as follows:"
  echo "  release/stable: ${0} ${release_stable}"
  echo "  release/latest: ${0} ${release_latest}"
  echo "  ci/latest:      ${0} ${ci_latest}"
}

function print-node-version-info() {
  echo "== $1 Node OS and Kubelet Versions =="
  "${KUBE_ROOT}/cluster/kubectl.sh" get nodes -o=jsonpath='{range .items[*]}name: "{.metadata.name}", osImage: "{.status.nodeInfo.osImage}", kubeletVersion: "{.status.nodeInfo.kubeletVersion}"{"\n"}{end}'
}

function upgrade-master() {
  local num_masters
  num_masters=$(get-master-replicas-count)
  if [[ "${num_masters}" -gt 1 ]]; then
    echo "Upgrade of master not supported if more than one master replica present. The current number of master replicas: ${num_masters}"
    exit 1
  fi

  echo "== Upgrading master to '${SERVER_BINARY_TAR_URL}'. Do not interrupt, deleting master instance. =="

  # Tries to figure out KUBE_USER/KUBE_PASSWORD by first looking under
  # kubeconfig:username, and then under kubeconfig:username-basic-auth.
  # TODO: KUBE_USER is used in generating ABAC policy which the
  # apiserver may not have enabled. If it's enabled, we must have a user
  # to generate a valid ABAC policy. If the username changes, should
  # the script fail? Should we generate a default username and password
  # if the section is missing in kubeconfig? Handle this better in 1.5.
  get-kubeconfig-basicauth
  get-kubeconfig-bearertoken

  detect-master
  parse-master-env
  upgrade-master-env

  # Delete the master instance. Note that the master-pd is created
  # with auto-delete=no, so it should not be deleted.
  gcloud compute instances delete \
    --project "${PROJECT}" \
    --quiet \
    --zone "${ZONE}" \
    "${MASTER_NAME}"

  create-master-instance "${MASTER_NAME}-ip"
  wait-for-master
}

function upgrade-master-env() {
  echo "== Upgrading master environment variables. =="
  # Generate the node problem detector token if it isn't present on the original
  # master.
 if [[ "${ENABLE_NODE_PROBLEM_DETECTOR:-}" == "standalone" && "${NODE_PROBLEM_DETECTOR_TOKEN:-}" == "" ]]; then
    NODE_PROBLEM_DETECTOR_TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
  fi
}

function wait-for-master() {
  echo "== Waiting for new master to respond to API requests =="

  local curl_auth_arg
  if [[ -n ${KUBE_BEARER_TOKEN:-} ]]; then
    curl_auth_arg=(-H "Authorization: Bearer ${KUBE_BEARER_TOKEN}")
  elif [[ -n ${KUBE_PASSWORD:-} ]]; then
    curl_auth_arg=(--user "${KUBE_USER}:${KUBE_PASSWORD}")
  else
    echo "can't get auth credentials for the current master"
    exit 1
  fi

  until curl --insecure "${curl_auth_arg[@]}" --max-time 5 \
    --fail --output /dev/null --silent "https://${KUBE_MASTER_IP}/healthz"; do
    printf "."
    sleep 2
  done

  echo "== Done =="
}

# Perform common upgrade setup tasks
#
# Assumed vars
#   KUBE_VERSION
function prepare-upgrade() {
  kube::util::ensure-temp-dir
  detect-project
  detect-subnetworks
  detect-node-names # sets INSTANCE_GROUPS
  write-cluster-location
  write-cluster-name
  tars_from_version
}

# Reads kube-env metadata from first node in NODE_NAMES.
#
# Assumed vars:
#   NODE_NAMES
#   PROJECT
#   ZONE
function get-node-env() {
  # TODO(zmerlynn): Make this more reliable with retries.
  gcloud compute --project ${PROJECT} ssh --zone ${ZONE} ${NODE_NAMES[0]} --command \
    "curl --fail --silent -H 'Metadata-Flavor: Google' \
      'http://metadata/computeMetadata/v1/instance/attributes/kube-env'" 2>/dev/null
}

# Read os distro information from /os/release on node.
# $1: The name of node
#
# Assumed vars:
#   PROJECT
#   ZONE
function get-node-os() {
  gcloud compute ssh "$1" \
    --project "${PROJECT}" \
    --zone "${ZONE}" \
    --command \
    "cat /etc/os-release | grep \"^ID=.*\" | cut -c 4-"
}

# Assumed vars:
#   KUBE_VERSION
#   NODE_SCOPES
#   NODE_INSTANCE_PREFIX
#   PROJECT
#   ZONE
#
# Vars set:
#   KUBE_PROXY_TOKEN
#   NODE_PROBLEM_DETECTOR_TOKEN
#   CA_CERT_BASE64
#   EXTRA_DOCKER_OPTS
#   KUBELET_CERT_BASE64
#   KUBELET_KEY_BASE64
function upgrade-nodes() {
  prepare-node-upgrade
  do-node-upgrade
}

function setup-base-image() {
  if [[ "${env_os_distro}" == "false" ]]; then
    echo "== Ensuring that new Node base OS image matched the existing Node base OS image"
    NODE_OS_DISTRIBUTION=$(get-node-os "${NODE_NAMES[0]}")

    if [[ "${NODE_OS_DISTRIBUTION}" == "cos" ]]; then
        NODE_OS_DISTRIBUTION="gci"
    fi
    
    source "${KUBE_ROOT}/cluster/gce/${NODE_OS_DISTRIBUTION}/node-helper.sh"
    # Reset the node image based on current os distro
    set-node-image
  fi
}

# prepare-node-upgrade creates a new instance template suitable for upgrading
# to KUBE_VERSION and echos a single line with the name of the new template.
#
# Assumed vars:
#   KUBE_VERSION
#   NODE_SCOPES
#   NODE_INSTANCE_PREFIX
#   PROJECT
#   ZONE
#
# Vars set:
#   SANITIZED_VERSION
#   INSTANCE_GROUPS
#   KUBE_PROXY_TOKEN
#   NODE_PROBLEM_DETECTOR_TOKEN
#   CA_CERT_BASE64
#   EXTRA_DOCKER_OPTS
#   KUBELET_CERT_BASE64
#   KUBELET_KEY_BASE64
function prepare-node-upgrade() {
  echo "== Preparing node upgrade (to ${KUBE_VERSION}). ==" >&2
  setup-base-image

  SANITIZED_VERSION=$(echo ${KUBE_VERSION} | sed 's/[\.\+]/-/g')

  # TODO(zmerlynn): Refactor setting scope flags.
  local scope_flags=
  if [ -n "${NODE_SCOPES}" ]; then
    scope_flags="--scopes ${NODE_SCOPES}"
  else
    scope_flags="--no-scopes"
  fi

  # Get required node env vars from exiting template.
  local node_env=$(get-node-env)
  KUBE_PROXY_TOKEN=$(get-env-val "${node_env}" "KUBE_PROXY_TOKEN")
  NODE_PROBLEM_DETECTOR_TOKEN=$(get-env-val "${node_env}" "NODE_PROBLEM_DETECTOR_TOKEN")
  CA_CERT_BASE64=$(get-env-val "${node_env}" "CA_CERT")
  EXTRA_DOCKER_OPTS=$(get-env-val "${node_env}" "EXTRA_DOCKER_OPTS")
  KUBELET_CERT_BASE64=$(get-env-val "${node_env}" "KUBELET_CERT")
  KUBELET_KEY_BASE64=$(get-env-val "${node_env}" "KUBELET_KEY")

  upgrade-node-env

  # TODO(zmerlynn): How do we ensure kube-env is written in a ${version}-
  #                 compatible way?
  write-node-env

  # TODO(zmerlynn): Get configure-vm script from ${version}. (Must plumb this
  #                 through all create-node-instance-template implementations).
  local template_name=$(get-template-name-from-version ${SANITIZED_VERSION})
  create-node-instance-template "${template_name}"
  # The following is echo'd so that callers can get the template name.
  echo "Instance template name: ${template_name}"
  echo "== Finished preparing node upgrade (to ${KUBE_VERSION}). ==" >&2
}

function upgrade-node-env() {
  echo "== Upgrading node environment variables. =="
  # Get the node problem detector token from master if it isn't present on
  # the original node.
  if [[ "${ENABLE_NODE_PROBLEM_DETECTOR:-}" == "standalone" && "${NODE_PROBLEM_DETECTOR_TOKEN:-}" == "" ]]; then
    detect-master
    local master_env=$(get-master-env)
    NODE_PROBLEM_DETECTOR_TOKEN=$(get-env-val "${master_env}" "NODE_PROBLEM_DETECTOR_TOKEN")
  fi
}

# Upgrades a single node.
# $1: The name of the node
#
# Note: This is called multiple times from do-node-upgrade() in parallel, so should be thread-safe.
function do-single-node-upgrade() {
  local -r instance="$1"
  local kubectl_rc
  local boot_id=$("${KUBE_ROOT}/cluster/kubectl.sh" get node "${instance}" --output=jsonpath='{.status.nodeInfo.bootID}' 2>&1) && kubectl_rc=$? || kubectl_rc=$?
  if [[ "${kubectl_rc}" != 0 ]]; then
    echo "== FAILED to get bootID ${instance} =="
    echo "${boot_id}"
    return ${kubectl_rc}
  fi

  # Drain node
  echo "== Draining ${instance}. == " >&2
  local drain_rc
  "${KUBE_ROOT}/cluster/kubectl.sh" drain --delete-local-data --force --ignore-daemonsets "${instance}" \
    && drain_rc=$? || drain_rc=$?
  if [[ "${drain_rc}" != 0 ]]; then
    echo "== FAILED to drain ${instance} =="
    return ${drain_rc}
  fi

  # Recreate instance
  echo "== Recreating instance ${instance}. ==" >&2
  local recreate_rc
  local recreate=$(gcloud compute instance-groups managed recreate-instances "${group}" \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    --instances="${instance}" 2>&1) && recreate_rc=$? || recreate_rc=$?
  if [[ "${recreate_rc}" != 0 ]]; then
    echo "== FAILED to recreate ${instance} =="
    echo "${recreate}"
    return ${recreate_rc}
  fi

  # Wait for node status to reflect a new boot ID. This guarantees us
  # that the node status in the API is from a different boot. This
  # does not guarantee that the status is from the upgraded node, but
  # it is a best effort approximation.
  echo "== Waiting for new node to be added to k8s.  ==" >&2
  while true; do
    local new_boot_id=$("${KUBE_ROOT}/cluster/kubectl.sh" get node "${instance}" --output=jsonpath='{.status.nodeInfo.bootID}' 2>&1) && kubectl_rc=$? || kubectl_rc=$?
    if [[ "${kubectl_rc}" != 0 ]]; then
      echo "== FAILED to get node ${instance} =="
      echo "${boot_id}"
      echo "  (Will retry.)"
    elif [[ "${boot_id}" != "${new_boot_id}" ]]; then
      echo "Node ${instance} recreated."
      break
    else
      echo -n .
    fi
    sleep 1
  done

  # Wait for the node to have Ready=True.
  echo "== Waiting for ${instance} to become ready. ==" >&2
  while true; do
    local ready=$("${KUBE_ROOT}/cluster/kubectl.sh" get node "${instance}" --output='jsonpath={.status.conditions[?(@.type == "Ready")].status}')
    if [[ "${ready}" != 'True' ]]; then
      echo "Node ${instance} is still not ready: Ready=${ready}"
    else
      echo "Node ${instance} Ready=${ready}"
      break
    fi
    sleep 1
  done

  # Uncordon the node.
  echo "== Uncordon ${instance}. == " >&2
  local uncordon_rc
  "${KUBE_ROOT}/cluster/kubectl.sh" uncordon "${instance}" \
    && uncordon_rc=$? || uncordon_rc=$?
  if [[ "${uncordon_rc}" != 0 ]]; then
    echo "== FAILED to uncordon ${instance} =="
    return ${uncordon_rc}
  fi
}

# Prereqs:
# - prepare-node-upgrade should have been called successfully
function do-node-upgrade() {
  echo "== Upgrading nodes to ${KUBE_VERSION} with max parallelism of ${node_upgrade_parallelism}. ==" >&2
  # Do the actual upgrade.
  # NOTE(zmerlynn): If you are changing this gcloud command, update
  #                 test/e2e/cluster_upgrade.go to match this EXACTLY.
  local template_name=$(get-template-name-from-version ${SANITIZED_VERSION})
  local old_templates=()
  local updates=()
  for group in ${INSTANCE_GROUPS[@]}; do
    old_templates+=($(gcloud compute instance-groups managed list \
        --project="${PROJECT}" \
        --filter="name ~ '${group}' AND zone:(${ZONE})" \
        --format='value(instanceTemplate)' || true))
    set_instance_template_out=$(gcloud compute instance-groups managed set-instance-template "${group}" \
      --template="${template_name}" \
      --project="${PROJECT}" \
      --zone="${ZONE}" 2>&1) && set_instance_template_rc=$? || set_instance_template_rc=$?
    if [[ "${set_instance_template_rc}" != 0 ]]; then
      echo "== FAILED to set-instance-template for ${group} to ${template_name} =="
      echo "${set_instance_template_out}"
      return ${set_instance_template_rc}
    fi
    instances=()
    instances+=($(gcloud compute instance-groups managed list-instances "${group}" \
        --format='value(instance)' \
        --project="${PROJECT}" \
        --zone="${ZONE}" 2>&1)) && list_instances_rc=$? || list_instances_rc=$?
    if [[ "${list_instances_rc}" != 0 ]]; then
      echo "== FAILED to list instances in group ${group} =="
      echo "${instances}"
      return ${list_instances_rc}
    fi

    process_count_left=${node_upgrade_parallelism}
    pids=()
    ret_code_sum=0  # Should stay 0 in the loop iff all parallel node upgrades succeed.
    for instance in ${instances[@]}; do
      do-single-node-upgrade "${instance}" & pids+=("$!")

      # We don't want to run more than ${node_upgrade_parallelism} upgrades at a time,
      # so wait once we hit that many nodes. This isn't ideal, since one might take much
      # longer than the others, but it should help.
      process_count_left=$((process_count_left - 1))
      if [[ process_count_left -eq 0 || "${instance}" == "${instances[-1]}" ]]; then
        # Wait for each of the parallel node upgrades to finish.
        for pid in "${pids[@]}"; do
          wait $pid
          ret_code_sum=$(( ret_code_sum + $? ))
        done
        # Return even if at least one of the node upgrades failed.
        if [[ ${ret_code_sum} != 0 ]]; then
          echo "== Some of the ${node_upgrade_parallelism} parallel node upgrades failed. =="
          return ${ret_code_sum}
        fi
        process_count_left=${node_upgrade_parallelism}
      fi
    done
  done

  # Remove the old templates.
  echo "== Deleting old templates in ${PROJECT}. ==" >&2
  for tmpl in ${old_templates[@]}; do
    gcloud compute instance-templates delete \
        --quiet \
        --project="${PROJECT}" \
        "${tmpl}" || true
  done

  echo "== Finished upgrading nodes to ${KUBE_VERSION}. ==" >&2
}

master_upgrade=true
node_upgrade=true
node_prereqs=false
local_binaries=false
env_os_distro=false
node_upgrade_parallelism=1

while getopts ":MNPlcho" opt; do
  case ${opt} in
    M)
      node_upgrade=false
      ;;
    N)
      master_upgrade=false
      ;;
    P)
      node_prereqs=true
      ;;
    l)
      local_binaries=true
      ;;
    c)
      node_upgrade_parallelism=${NODE_UPGRADE_PARALLELISM:-1}
      ;;
    o)
      env_os_distro=true
      ;;
    h)
      usage
      exit 0
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      usage
      exit 1
      ;;
  esac
done
shift $((OPTIND-1))

if [[ $# -gt 1 ]]; then
  echo "Error: Only one parameter (<version number or publication>) may be passed after the set of flags!" >&2
  usage
  exit 1
fi

if [[ $# -lt 1 ]] && [[ "${local_binaries}" == "false" ]]; then
  usage
  exit 1
fi

if [[ "${master_upgrade}" == "false" ]] && [[ "${node_upgrade}" == "false" ]]; then
  echo "Can't specify both -M and -N" >&2
  exit 1
fi

# prompt if etcd storage media type isn't set unless using etcd2 when doing master upgrade
if [[ -z "${STORAGE_MEDIA_TYPE:-}" ]] && [[ "${STORAGE_BACKEND:-}" != "etcd2" ]] && [[ "${master_upgrade}" == "true" ]]; then
  echo "The default etcd storage media type in 1.6 has changed from application/json to application/vnd.kubernetes.protobuf."
  echo "Documentation about the change can be found at https://kubernetes.io/docs/admin/etcd_upgrade."
  echo ""
  echo "ETCD2 DOES NOT SUPPORT PROTOBUF: If you wish to have to ability to downgrade to etcd2 later application/json must be used."
  echo ""
  echo "It's HIGHLY recommended that etcd be backed up before this step!!"
  echo ""
  echo "To enable using json, before running this script set:"
  echo "export STORAGE_MEDIA_TYPE=application/json"
  echo ""
  if [ -t 0 ] && [ -t 1 ]; then
    read -p "Would you like to continue with the new default, and lose the ability to downgrade to etcd2? [y/N] " confirm
    if [[ "${confirm}" != "y" ]]; then
      exit 1
    fi
  else
    echo "To enable using protobuf, before running this script set:"
    echo "export STORAGE_MEDIA_TYPE=application/vnd.kubernetes.protobuf"
    echo ""
    echo "STORAGE_MEDIA_TYPE must be specified when run non-interactively." >&2
    exit 1
  fi
fi

# Prompt if etcd image/version is unspecified when doing master upgrade.
# In e2e tests, we use TEST_ALLOW_IMPLICIT_ETCD_UPGRADE=true to skip this
# prompt, simulating the behavior when the user confirms interactively.
# All other automated use of this script should explicitly specify a version.
if [[ "${master_upgrade}" == "true" ]]; then
  if [[ -z "${ETCD_IMAGE:-}" && -z "${TEST_ETCD_IMAGE:-}" ]] || [[ -z "${ETCD_VERSION:-}" && -z "${TEST_ETCD_VERSION:-}" ]]; then
    echo
    echo "***WARNING***"
    echo "Upgrading Kubernetes with this script might result in an upgrade to a new etcd version."
    echo "Some etcd version upgrades, such as 3.0.x to 3.1.x, DO NOT offer a downgrade path."
    echo "To pin the etcd version to your current one (e.g. v3.0.17), set the following variables"
    echo "before running this script:"
    echo
    echo "# example: pin to etcd v3.0.17"
    echo "export ETCD_IMAGE=3.0.17"
    echo "export ETCD_VERSION=3.0.17"
    echo
    echo "Alternatively, if you choose to allow an etcd upgrade that doesn't support downgrade,"
    echo "you might still be able to downgrade Kubernetes by pinning to the newer etcd version."
    echo "In all cases, it is strongly recommended to have an etcd backup before upgrading."
    echo
    if [ -t 0 ] && [ -t 1 ]; then
      read -p "Continue with default etcd version, which might upgrade etcd? [y/N] " confirm
      if [[ "${confirm}" != "y" ]]; then
        exit 1
      fi
    elif [[ "${TEST_ALLOW_IMPLICIT_ETCD_UPGRADE:-}" != "true" ]]; then
      echo "ETCD_IMAGE and ETCD_VERSION must be specified when run non-interactively." >&2
      exit 1
    fi
  fi
fi

print-node-version-info "Pre-Upgrade"

if [[ "${local_binaries}" == "false" ]]; then
  set_binary_version ${1}
fi

prepare-upgrade

if [[ "${node_prereqs}" == "true" ]]; then
  prepare-node-upgrade
  exit 0
fi

if [[ "${master_upgrade}" == "true" ]]; then
  upgrade-master
fi

if [[ "${node_upgrade}" == "true" ]]; then
  if [[ "${local_binaries}" == "true" ]]; then
    echo "Upgrading nodes to local binaries is not yet supported." >&2
    exit 1
  else
    upgrade-nodes
  fi
fi

echo "== Validating cluster post-upgrade =="
"${KUBE_ROOT}/cluster/validate-cluster.sh"

print-node-version-info "Post-Upgrade"
