#!/bin/bash

# Copyright 2014 Google Inc. All rights reserved.
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
KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/cluster/gce/${KUBE_CONFIG_FILE-"config-default.sh"}"

# Verify prereqs
function verify-prereqs {
  local cmd
  for cmd in gcloud gsutil; do
    which "${cmd}" >/dev/null || {
      echo "Can't find ${cmd} in PATH, please fix and retry. The Google Cloud "
      echo "SDK can be downloaded from https://cloud.google.com/sdk/."
      exit 1
    }
  done
}

# Create a temp dir that'll be deleted at the end of this bash session.
#
# Vars set:
#   KUBE_TEMP
function ensure-temp-dir {
  if [[ -z ${KUBE_TEMP-} ]]; then
    KUBE_TEMP=$(mktemp -d -t kubernetes.XXXXXX)
    trap 'rm -rf "${KUBE_TEMP}"' EXIT
  fi
}

# Verify and find the various tar files that we are going to use on the server.
#
# Vars set:
#   SERVER_BINARY_TAR
#   SALT_TAR
function find-release-tars {
  SERVER_BINARY_TAR="${KUBE_ROOT}/server/kubernetes-server-linux-amd64.tar.gz"
  if [[ ! -f "$SERVER_BINARY_TAR" ]]; then
    SERVER_BINARY_TAR="${KUBE_ROOT}/_output/release-tars/kubernetes-server-linux-amd64.tar.gz"
  fi
  if [[ ! -f "$SERVER_BINARY_TAR" ]]; then
    echo "!!! Cannot find kubernetes-server-linux-amd64.tar.gz"
    exit 1
  fi

  SALT_TAR="${KUBE_ROOT}/server/kubernetes-salt.tar.gz"
  if [[ ! -f "$SALT_TAR" ]]; then
    SALT_TAR="${KUBE_ROOT}/_output/release-tars/kubernetes-salt.tar.gz"
  fi
  if [[ ! -f "$SALT_TAR" ]]; then
    echo "!!! Cannot find kubernetes-salt.tar.gz"
    exit 1
  fi
}

# Use the gcloud defaults to find the project.  If it is already set in the
# environment then go with that.
#
# Vars set:
#   PROJECT
function detect-project () {
  if [[ -z "${PROJECT-}" ]]; then
    PROJECT=$(gcloud config list project | tail -n 1 | cut -f 3 -d ' ')
  fi

  if [[ -z "${PROJECT-}" ]]; then
    echo "Could not detect Google Cloud Platform project.  Set the default project using " >&2
    echo "'gcloud config set project <PROJECT>'" >&2
    exit 1
  fi
  echo "Project: $PROJECT (autodetected from gcloud config)"
}


# Take the local tar files and upload them to Google Storage.  They will then be
# downloaded by the master as part of the start up script for the master.
#
# Assumed vars:
#   PROJECT
#   SERVER_BINARY_TAR
#   SALT_TAR
# Vars set:
#   SERVER_BINARY_TAR_URL
#   SALT_TAR_URL
function upload-server-tars() {
  SERVER_BINARY_TAR_URL=
  SALT_TAR_URL=

  local project_hash
  if which md5 > /dev/null 2>&1; then
    project_hash=$(md5 -q -s "$PROJECT")
  else
    project_hash=$(echo -n "$PROJECT" | md5sum | awk '{ print $1 }')
  fi
  project_hash=${project_hash:0:5}

  local -r staging_bucket="gs://kubernetes-staging-${project_hash}"

  # Ensure the bucket is created
  if ! gsutil ls "$staging_bucket" > /dev/null 2>&1 ; then
    echo "Creating $staging_bucket"
    gsutil mb "${staging_bucket}"
  fi

  local -r staging_path="${staging_bucket}/devel"

  echo "+++ Staging server tars to Google Storage: ${staging_path}"
  local server_binary_gs_url="${staging_path}/${SERVER_BINARY_TAR##*/}"
  gsutil -q -h "Cache-Control:private, max-age=0" cp "${SERVER_BINARY_TAR}" "${server_binary_gs_url}"
  gsutil acl ch -g all:R "${server_binary_gs_url}" >/dev/null 2>&1
  local salt_gs_url="${staging_path}/${SALT_TAR##*/}"
  gsutil -q -h "Cache-Control:private, max-age=0" cp "${SALT_TAR}" "${salt_gs_url}"
  gsutil acl ch -g all:R "${salt_gs_url}" >/dev/null 2>&1

  # Convert from gs:// URL to an https:// URL
  SERVER_BINARY_TAR_URL="${server_binary_gs_url/gs:\/\//https://storage.googleapis.com/}"
  SALT_TAR_URL="${salt_gs_url/gs:\/\//https://storage.googleapis.com/}"
}

# Detect the information about the minions
#
# Assumed vars:
#   MINION_NAMES
#   ZONE
# Vars set:
#   KUBE_MINION_IP_ADDRESS (array)
function detect-minions () {
  if [[ -z "${PROJECT-}" ]]; then
    detect-project
  fi
  KUBE_MINION_IP_ADDRESSES=()
  for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
    local minion_ip=$(gcloud compute instances describe --project "${PROJECT}" --zone "${ZONE}" \
      "${MINION_NAMES[$i]}" --fields networkInterfaces[0].accessConfigs[0].natIP \
      --format=text | awk '{ print $2 }')
    if [[ -z "${minion_ip-}" ]] ; then
      echo "Did not find ${MINION_NAMES[$i]}" >&2
    else
      echo "Found ${MINION_NAMES[$i]} at ${minion_ip}"
      KUBE_MINION_IP_ADDRESSES+=("${minion_ip}")
    fi
  done
  if [[ -z "${KUBE_MINION_IP_ADDRESSES-}" ]]; then
    echo "Could not detect Kubernetes minion nodes.  Make sure you've launched a cluster with 'kube-up.sh'" >&2
    exit 1
  fi
}

# Detect the IP for the master
#
# Assumed vars:
#   MASTER_NAME
#   PROJECT (if unset, will detect-project)
#   ZONE
# Vars set:
#   KUBE_MASTER
#   KUBE_MASTER_IP
function detect-master () {
  if [[ -z "${PROJECT-}" ]]; then
    detect-project
  fi
  KUBE_MASTER=${MASTER_NAME}
  if [[ -z "${KUBE_MASTER_IP-}" ]]; then
    KUBE_MASTER_IP=$(gcloud compute instances describe --project "${PROJECT}" --zone "${ZONE}" \
      "${MASTER_NAME}" --fields networkInterfaces[0].accessConfigs[0].natIP \
      --format=text | awk '{ print $2 }')
  fi
  if [[ -z "${KUBE_MASTER_IP-}" ]]; then
    echo "Could not detect Kubernetes master node.  Make sure you've launched a cluster with 'kube-up.sh'" >&2
    exit 1
  fi
  echo "Using master: $KUBE_MASTER (external IP: $KUBE_MASTER_IP)"
}

# Ensure that we have a password created for validating to the master.  Will
# read from $HOME/.kubernetres_auth if available.
#
# Vars set:
#   KUBE_USER
#   KUBE_PASSWORD
function get-password {
  local file="$HOME/.kubernetes_auth"
  if [[ -r "$file" ]]; then
    KUBE_USER=$(cat "$file" | python -c 'import json,sys;print json.load(sys.stdin)["User"]')
    KUBE_PASSWORD=$(cat "$file" | python -c 'import json,sys;print json.load(sys.stdin)["Password"]')
    return
  fi
  KUBE_USER=admin
  KUBE_PASSWORD=$(python -c 'import string,random; print "".join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(16))')

  # Remove this code, since in all use cases I can see, we are overwriting this
  # at cluster creation time.
  cat << EOF > "$file"
{
  "User": "$KUBE_USER",
  "Password": "$KUBE_PASSWORD"
}
EOF
  chmod 0600 "$file"
}

# Generate authentication token for admin user. Will
# read from $HOME/.kubernetes_auth if available.
#
# Vars set:
#   KUBE_ADMIN_TOKEN
function get-admin-token {
  local file="$HOME/.kubernetes_auth"
  if [[ -r "$file" ]]; then
    KUBE_ADMIN_TOKEN=$(cat "$file" | python -c 'import json,sys;print json.load(sys.stdin)["BearerToken"]')
    return
  fi
  KUBE_ADMIN_TOKEN=$(python -c 'import string,random; print "".join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(32))')
}



# Wait for background jobs to finish. Exit with
# an error status if any of the jobs failed.
function wait-for-jobs {
  local fail=0
  local job
  for job in $(jobs -p); do
    wait "${job}" || fail=$((fail + 1))
  done
  if (( fail != 0 )); then
    echo -e "${color_red}${fail} commands failed.  Exiting.${color_norm}" >&2
    # Ignore failures for now.
    # exit 2
  fi
}

# Robustly try to create a firewall rule.
# $1: The name of firewall rule.
# $2: IP ranges.
# $3: Target tags for this firewall rule.
function create-firewall-rule {
  local attempt=0
  while true; do
    if ! gcloud compute firewall-rules create "$1" \
      --project "${PROJECT}" \
      --network "${NETWORK}" \
      --source-ranges "$2" \
      --target-tags "$3" \
      --allow tcp udp icmp esp ah sctp; then 
        if (( attempt > 5 )); then
          echo -e "${color_red}Failed to create firewall rule $1 ${color_norm}"
          exit 2
        fi
        echo -e "${color_yellow}Attempt $(($attempt+1)) failed to create firewall rule $1. Retrying.${color_norm}"
        attempt=$(($attempt+1))
    else
       break
    fi
  done
}

# Robustly try to create a route.
# $1: The name of the route.
# $2: IP range.
function create-route {
  local attempt=0
  while true; do
    if ! gcloud compute routes create "$1" \
      --project "${PROJECT}" \
      --destination-range "$2" \
      --network "${NETWORK}" \
      --next-hop-instance "$1" \
      --next-hop-instance-zone "${ZONE}"; then
        if (( attempt > 5 )); then
          echo -e "${color_red}Failed to create route $1 ${color_norm}"
          exit 2
        fi
        echo -e "${color_yellow}Attempt $(($attempt+1)) failed to create route $1. Retrying.${color_norm}"
        attempt=$(($attempt+1))
    else
       break
    fi
  done
}

# Robustly try to create an instance.
# $1: The name of the instance.
# $2: The scopes flag.
# $3: The minion start script.
function create-instance {
  local attempt=0
  while true; do
    if ! gcloud compute instances create "$1" \
      --project "${PROJECT}" \
      --zone "${ZONE}" \
      --machine-type "${MINION_SIZE}" \
      --image-project="${IMAGE_PROJECT}" \
      --image "${IMAGE}" \
      --tags "${MINION_TAG}" \
      --network "${NETWORK}" \
      $2 \
      --can-ip-forward \
      --metadata-from-file "$3"; then
        if (( attempt > 5 )); then
          echo -e "${color_red}Failed to create instance $1 ${color_norm}"
          exit 2
        fi
        echo -e "${color_yellow}Attempt $(($attempt+1)) failed to create node $1. Retrying.${color_norm}"
        attempt=$(($attempt+1))
        # Attempt to delete the disk for this node (the disk may have been created even
        # if the instance creation failed).
        gcloud compute disks delete "$1" --project "${PROJECT}" --zone "${ZONE}" --quiet || true
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
function kube-up {
  # Detect the project into $PROJECT if it isn't set
  detect-project

  # Make sure we have the tar files staged on Google Storage
  find-release-tars
  upload-server-tars

  ensure-temp-dir

  get-password
  python "${KUBE_ROOT}/third_party/htpasswd/htpasswd.py" \
    -b -c "${KUBE_TEMP}/htpasswd" "$KUBE_USER" "$KUBE_PASSWORD"
  local htpasswd
  htpasswd=$(cat "${KUBE_TEMP}/htpasswd")

  if ! gcloud compute networks describe "${NETWORK}" &>/dev/null; then
    echo "Creating new network: ${NETWORK}"
    # The network needs to be created synchronously or we have a race. The
    # firewalls can be added concurrent with instance creation.
    gcloud compute networks create "${NETWORK}" --range "10.240.0.0/16"
  fi

  if ! gcloud compute firewall-rules describe "${NETWORK}-default-internal" &>/dev/null; then
    gcloud compute firewall-rules create "${NETWORK}-default-internal" \
      --project "${PROJECT}" \
      --network "${NETWORK}" \
      --source-ranges "10.0.0.0/8" \
      --allow "tcp:1-65535" "udp:1-65535" "icmp" &
  fi

  if ! gcloud compute firewall-rules describe "${NETWORK}-default-ssh" &>/dev/null; then
    gcloud compute firewall-rules create "${NETWORK}-default-ssh" \
      --project "${PROJECT}" \
      --network "${NETWORK}" \
      --source-ranges "0.0.0.0/0" \
      --allow "tcp:22" &
  fi

  echo "Starting VMs and configuring firewalls"
  gcloud compute firewall-rules create "${MASTER_NAME}-https" \
    --project "${PROJECT}" \
    --network "${NETWORK}" \
    --target-tags "${MASTER_TAG}" \
    --allow tcp:443 &

  (
    echo "#! /bin/bash"
    echo "mkdir -p /var/cache/kubernetes-install"
    echo "cd /var/cache/kubernetes-install"
    echo "readonly MASTER_NAME='${MASTER_NAME}'"
    echo "readonly NODE_INSTANCE_PREFIX='${INSTANCE_PREFIX}-minion'"
    echo "readonly SERVER_BINARY_TAR_URL='${SERVER_BINARY_TAR_URL}'"
    echo "readonly SALT_TAR_URL='${SALT_TAR_URL}'"
    echo "readonly MASTER_HTPASSWD='${htpasswd}'"
    echo "readonly PORTAL_NET='${PORTAL_NET}'"
    echo "readonly ENABLE_NODE_MONITORING='${ENABLE_NODE_MONITORING:-false}'"
    echo "readonly ENABLE_NODE_LOGGING='${ENABLE_NODE_LOGGING:-false}'"
    echo "readonly LOGGING_DESTINATION='${LOGGING_DESTINATION:-}'"
    grep -v "^#" "${KUBE_ROOT}/cluster/gce/templates/common.sh"
    grep -v "^#" "${KUBE_ROOT}/cluster/gce/templates/create-dynamic-salt-files.sh"
    grep -v "^#" "${KUBE_ROOT}/cluster/gce/templates/download-release.sh"
    grep -v "^#" "${KUBE_ROOT}/cluster/gce/templates/salt-master.sh"
  ) > "${KUBE_TEMP}/master-start.sh"

  # Report logging choice (if any).
  if [[ "${ENABLE_NODE_LOGGING-}" == "true" ]]; then
    echo "+++ Logging using Fluentd to ${LOGGING_DESTINATION:-unknown}"
    # For logging to GCP we need to enable some minion scopes.
    if [[ "${LOGGING_DESTINATION-}" == "gcp" ]]; then
      MINION_SCOPES+=('https://www.googleapis.com/auth/logging.write')
    fi
  fi

  gcloud compute instances create "${MASTER_NAME}" \
    --project "${PROJECT}" \
    --zone "${ZONE}" \
    --machine-type "${MASTER_SIZE}" \
    --image-project="${IMAGE_PROJECT}" \
    --image "${IMAGE}" \
    --tags "${MASTER_TAG}" \
    --network "${NETWORK}" \
    --scopes "storage-ro" "compute-rw" \
    --metadata-from-file "startup-script=${KUBE_TEMP}/master-start.sh" &

  # Create a single firewall rule for all minions.
  create-firewall-rule "${MINION_TAG}-all" "${CLUSTER_IP_RANGE}" "${MINION_TAG}" &

  # Wait for last batch of jobs.
  wait-for-jobs

  # Create the routes, 10 at a time.
  for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
    create-route "${MINION_NAMES[$i]}" "${MINION_IP_RANGES[$i]}" &

    if [ $i -ne 0 ] && [ $((i%10)) -eq 0 ]; then
      echo Waiting for a batch of routes at $i...
      wait-for-jobs
    fi

  done
  # Wait for last batch of jobs.
  wait-for-jobs

  local -a scope_flags=()
  if (( "${#MINION_SCOPES[@]}" > 0 )); then
    scope_flags=("--scopes" "${MINION_SCOPES[@]}")
  else
    scope_flags=("--no-scopes")
  fi
  # Create the instances, 5 at a time.
  for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
    (
      echo "#! /bin/bash"
      echo "ZONE='${ZONE}'"
      echo "MASTER_NAME='${MASTER_NAME}'"
      echo "MINION_IP_RANGE='${MINION_IP_RANGES[$i]}'"
      echo "EXTRA_DOCKER_OPTS='${EXTRA_DOCKER_OPTS}'"
      echo "ENABLE_DOCKER_REGISTRY_CACHE='${ENABLE_DOCKER_REGISTRY_CACHE:-false}'"
      grep -v "^#" "${KUBE_ROOT}/cluster/gce/templates/common.sh"
      grep -v "^#" "${KUBE_ROOT}/cluster/gce/templates/salt-minion.sh"
    ) > "${KUBE_TEMP}/minion-start-${i}.sh"

    local scopes_flag="${scope_flags[@]}"
    create-instance "${MINION_NAMES[$i]}" "${scopes_flag}" "startup-script=${KUBE_TEMP}/minion-start-${i}.sh" &

    if [ $i -ne 0 ] && [ $((i%5)) -eq 0 ]; then
      echo Waiting for creation of a batch of instances at $i...
      wait-for-jobs
    fi

  done
  # Wait for last batch of jobs.
  wait-for-jobs

  detect-master

  echo "Waiting for cluster initialization."
  echo
  echo "  This will continually check to see if the API for kubernetes is reachable."
  echo "  This might loop forever if there was some uncaught error during start"
  echo "  up."
  echo

  until curl --insecure --user "${KUBE_USER}:${KUBE_PASSWORD}" --max-time 5 \
          --fail --output /dev/null --silent "https://${KUBE_MASTER_IP}/api/v1beta1/pods"; do
      printf "."
      sleep 2
  done

  echo "Kubernetes cluster created."

  local kube_cert=".kubecfg.crt"
  local kube_key=".kubecfg.key"
  local ca_cert=".kubernetes.ca.crt"

  # TODO: generate ADMIN (and KUBELET) tokens and put those in the master's
  # config file.  Distribute the same way the htpasswd is done.
  (umask 077
   gcloud compute ssh --project "${PROJECT}" --zone "$ZONE" "${MASTER_NAME}" --command "sudo cat /srv/kubernetes/kubecfg.crt" >"${HOME}/${kube_cert}" 2>/dev/null
   gcloud compute ssh --project "${PROJECT}" --zone "$ZONE" "${MASTER_NAME}" --command "sudo cat /srv/kubernetes/kubecfg.key" >"${HOME}/${kube_key}" 2>/dev/null
   gcloud compute ssh --project "${PROJECT}" --zone "$ZONE" "${MASTER_NAME}" --command "sudo cat /srv/kubernetes/ca.crt" >"${HOME}/${ca_cert}" 2>/dev/null

   cat << EOF > ~/.kubernetes_auth
{
  "User": "$KUBE_USER",
  "Password": "$KUBE_PASSWORD",
  "CAFile": "$HOME/$ca_cert",
  "CertFile": "$HOME/$kube_cert",
  "KeyFile": "$HOME/$kube_key"
}
EOF

   chmod 0600 ~/.kubernetes_auth "${HOME}/${kube_cert}" \
     "${HOME}/${kube_key}" "${HOME}/${ca_cert}"
   echo Wrote ~/.kubernetes_auth
  )

  echo "Sanity checking cluster..."

  # Basic sanity checking
  local i
  local rc # Capture return code without exiting because of errexit bash option
  for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
      # Make sure docker is installed and working.
      local attempt=0
      while true; do
        echo -n Attempt "$(($attempt+1))" to check Docker on node "${MINION_NAMES[$i]}" ...
        local output=$(gcloud compute --project "${PROJECT}" ssh --zone "$ZONE" "${MINION_NAMES[$i]}" --command "sudo docker ps -a" 2>/dev/null)
        if [[ -z "${output}" ]]; then
          if (( attempt > 9 )); then
            echo
            echo -e "${color_red}Docker failed to install on node ${MINION_NAMES[$i]}. Your cluster is unlikely" >&2
            echo "to work correctly. Please run ./cluster/kube-down.sh and re-create the" >&2
            echo -e "cluster. (sorry!)${color_norm}" >&2
            exit 1
          fi
        elif [[ "${output}" != *"kubernetes/pause"* ]]; then
          if (( attempt > 9 )); then
            echo
            echo -e "${color_red}Failed to observe kubernetes/pause on node ${MINION_NAMES[$i]}. Your cluster is unlikely" >&2
            echo "to work correctly. Please run ./cluster/kube-down.sh and re-create the" >&2
            echo -e "cluster. (sorry!)${color_norm}" >&2
            exit 1
          fi
        else
          echo -e " ${color_green}[working]${color_norm}"
          break
        fi
        echo -e " ${color_yellow}[not working yet]${color_norm}"
        # Start Docker, in case it failed to start.
        gcloud compute --project "${PROJECT}" ssh --zone "$ZONE" "${MINION_NAMES[$i]}" \
                       --command "sudo service docker start" 2>/dev/null || true
        attempt=$(($attempt+1))
        sleep 30
      done
  done

  echo
  echo -e "${color_green}Kubernetes cluster is running.  The master is running at:"
  echo
  echo -e "${color_yellow}  https://${KUBE_MASTER_IP}"
  echo
  echo -e "${color_green}The user name and password to use is located in ~/.kubernetes_auth.${color_norm}"
  echo

}

# Delete a kubernetes cluster.
#
# Assumed vars:
#   MASTER_NAME
#   INSTANCE_PREFIX
#   ZONE
#   PROJECT
# This function tears down cluster resources 10 at a time to avoid issuing too many
# API calls and exceeding API quota. It is important to bring down the instances before bringing
# down the firewall rules and routes.
function kube-down {
  # Detect the project into $PROJECT
  detect-project

  echo "Bringing down cluster"

  # First delete the master (if it exists).
  gcloud compute instances delete \
    --project "${PROJECT}" \
    --quiet \
    --delete-disks all \
    --zone "${ZONE}" \
    "${MASTER_NAME}" || true
  # Find out what minions are running.
  local -a minions
  minions=( $(gcloud compute instances list \
                --project "${PROJECT}" --zone "${ZONE}" \
                --regexp "${INSTANCE_PREFIX}-minion-[0-9]+" \
                | awk 'NR >= 2 { print $1 }') )
  # If any minions are running, delete them in batches.
  while (( "${#minions[@]}" > 0 )); do
    echo Deleting nodes "${minions[*]::10}"
    gcloud compute instances delete \
      --project "${PROJECT}" \
      --quiet \
      --delete-disks boot \
      --zone "${ZONE}" \
      "${minions[@]::10}" || true
    minions=( "${minions[@]:10}" )
  done

  # Delete firewall rule for the master.
  gcloud compute firewall-rules delete  \
    --project "${PROJECT}" \
    --quiet \
    "${MASTER_NAME}-https" || true

  # Delete firewall rule for minions.
  gcloud compute firewall-rules delete  \
    --project "${PROJECT}" \
    --quiet \
    "${MINION_TAG}-all" || true

  # Delete routes.
  local -a routes
  routes=( $(gcloud compute routes list --project "${PROJECT}" \
              --regexp "${INSTANCE_PREFIX}-minion-[0-9]+" | awk 'NR >= 2 { print $1 }') )
  while (( "${#routes[@]}" > 0 )); do
    echo Deleting routes "${routes[*]::10}"
    gcloud compute routes delete \
      --project "${PROJECT}" \
      --quiet \
      "${routes[@]::10}" || true
    routes=( "${routes[@]:10}" )
  done

}

# Update a kubernetes cluster with latest source
function kube-push {
  detect-project
  detect-master

  # Make sure we have the tar files staged on Google Storage
  find-release-tars
  upload-server-tars

  (
    echo "#! /bin/bash"
    echo "mkdir -p /var/cache/kubernetes-install"
    echo "cd /var/cache/kubernetes-install"
    echo "readonly SERVER_BINARY_TAR_URL='${SERVER_BINARY_TAR_URL}'"
    echo "readonly SALT_TAR_URL='${SALT_TAR_URL}'"
    grep -v "^#" "${KUBE_ROOT}/cluster/gce/templates/common.sh"
    grep -v "^#" "${KUBE_ROOT}/cluster/gce/templates/download-release.sh"
    echo "echo Executing configuration"
    echo "sudo salt '*' mine.update"
    echo "sudo salt --force-color '*' state.highstate"
  ) | gcloud compute ssh --project "${PROJECT}" --zone "$ZONE" "$KUBE_MASTER" --command "sudo bash"

  get-password

  echo
  echo "Kubernetes cluster is running.  The master is running at:"
  echo
  echo "  https://${KUBE_MASTER_IP}"
  echo
  echo "The user name and password to use is located in ~/.kubernetes_auth."
  echo

}

# -----------------------------------------------------------------------------
# Cluster specific test helpers used from hack/e2e-test.sh

# Execute prior to running tests to build a release if required for env.
#
# Assumed Vars:
#   KUBE_ROOT
function test-build-release {
  # Make a release
  "${KUBE_ROOT}/build/release.sh"
}

# Execute prior to running tests to initialize required structure. This is
# called from hack/e2e-test.sh.
#
# Assumed vars:
#   PROJECT
#   Variables from config.sh
function test-setup {

  # Detect the project into $PROJECT if it isn't set
  # gce specific
  detect-project

  # Open up port 80 & 8080 so common containers on minions can be reached
  gcloud compute firewall-rules create \
    --project "${PROJECT}" \
    --target-tags "${MINION_TAG}" \
    --allow tcp:80 tcp:8080 \
    --network "${NETWORK}" \
    "${MINION_TAG}-${INSTANCE_PREFIX}-http-alt"
}

# Execute after running tests to perform any required clean-up.  This is called
# from hack/e2e-test.sh
#
# Assumed Vars:
#   PROJECT
function test-teardown {
  echo "Shutting down test cluster in background."
  gcloud compute firewall-rules delete  \
    --project "${PROJECT}" \
    --quiet \
    "${MINION_TAG}-${INSTANCE_PREFIX}-http-alt" || true
  "${KUBE_ROOT}/cluster/kube-down.sh"
}

# SSH to a node by name ($1) and run a command ($2).
function ssh-to-node {
  local node="$1"
  local cmd="$2"
  gcloud compute ssh --ssh-flag="-o LogLevel=quiet" --zone="${ZONE}" "${node}" --command "${cmd}"
}

# Restart the kube-proxy on a node ($1)
function restart-kube-proxy {
  ssh-to-node "$1" "sudo /etc/init.d/kube-proxy restart"
}

# Setup monitoring using heapster and InfluxDB
function setup-monitoring {
  if [[ "${ENABLE_CLUSTER_MONITORING}" == "true" ]]; then
    echo "Setting up cluster monitoring using Heapster."

    if ! gcloud compute firewall-rules describe monitoring-heapster &>/dev/null; then
      if ! gcloud compute firewall-rules create monitoring-heapster \
          --project "${PROJECT}" \
          --target-tags="${MINION_TAG}" \
          --allow tcp:80 tcp:8083 tcp:8086 tcp:9200; then
        echo "Failed to set up firewall for monitoring" && false
      fi
    fi

    # Re-use master auth for Grafana
    get-password
    ensure-temp-dir

    cp "${KUBE_ROOT}/examples/monitoring/influx-grafana-pod.json" "${KUBE_TEMP}/influx-grafana-pod.0.json"
    sed "s/HTTP_USER, \"value\": \"[^\"]*\"/HTTP_USER, \"value\": \"$KUBE_USER\"/g" \
        "${KUBE_TEMP}/influx-grafana-pod.0.json" > "${KUBE_TEMP}/influx-grafana-pod.1.json"
    sed "s/HTTP_PASS, \"value\": \"[^\"]*\"/HTTP_PASS, \"value\": \"$KUBE_PASSWORD\"/g" \
        "${KUBE_TEMP}/influx-grafana-pod.1.json" > "${KUBE_TEMP}/influx-grafana-pod.2.json"
    local kubectl="${KUBE_ROOT}/cluster/kubectl.sh"
    if "${kubectl}" create -f "${KUBE_TEMP}/influx-grafana-pod.2.json" &> /dev/null \
        && "${kubectl}" create -f "${KUBE_ROOT}/examples/monitoring/influx-grafana-service.json" &> /dev/null \
        && "${kubectl}" create -f "${KUBE_ROOT}/examples/monitoring/heapster-pod.json" &> /dev/null; then
      local dashboard_url="http://$(${kubectl} get -o json pod influx-grafana | grep hostIP | awk '{print $2}' | sed 's/[,|\"]//g')"
      echo
      echo "Grafana dashboard will be available at $dashboard_url. Wait for the monitoring dashboard to be online."
      echo "Use the master user name and password for the dashboard."
      echo
    else
      echo "Failed to Setup Monitoring"
      teardown-monitoring
    fi
  fi
}

function teardown-monitoring {
  if [[ "${ENABLE_CLUSTER_MONITORING}" == "true" ]]; then
    detect-project

    local kubectl="${KUBE_ROOT}/cluster/kubectl.sh"
    "${kubectl}" delete pods heapster &> /dev/null || true
    "${kubectl}" delete pods influx-grafana &> /dev/null || true
    "${kubectl}" delete services influx-master &> /dev/null || true
    if gcloud compute firewall-rules describe monitoring-heapster &> /dev/null; then
      gcloud compute firewall-rules delete \
          --project "${PROJECT}" \
          --quiet \
          monitoring-heapster &> /dev/null || true
    fi
  fi
}

# Perform preparations required to run e2e tests
function prepare-e2e() {
  detect-project
}
