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

# Set the aws cli tool to output text (not json!)
export AWS_DEFAULT_OUTPUT="text"

# Use the config file specified in $KUBE_CONFIG_FILE, or default to
# config-default.sh.
KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/cluster/ec2/${KUBE_CONFIG_FILE-"config-default.sh"}"

# Verify prereqs
function verify-prereqs {
  local cmd
  for cmd in aws; do
    which "${cmd}" >/dev/null || {
      echo "Can't find ${cmd} in PATH, please fix and retry."
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

# Use the aws defaults to find the project.  If it is already set in the
# environment then go with that.
#
# Vars set:
#   PROJECT
function detect-project () {
  # TODO: Do we need a project? Should this be == CLUSTER_KEY?
  PROJECT=""
#  if [[ -z "${PROJECT-}" ]]; then
#    PROJECT=$(aws config list project | tail -n 1 | cut -f 3 -d ' ')
#  fi
#
#  if [[ -z "${PROJECT-}" ]]; then
#    echo "Could not detect Google Cloud Platform project.  Set the default project using " >&2
#    echo "'aws config set project <PROJECT>'" >&2
#    exit 1
#  fi
#  echo "Project: $PROJECT (autodetected from gcloud config)"
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
    project_hash=$(echo -n "$PROJECT" | md5sum)
  fi
  project_hash=${project_hash:0:6}

  local -r staging_bucket="kubernetes-staging-${project_hash}"

  # Ensure the bucket is created
  if ! aws s3 ls "s3://$staging_bucket" > /dev/null 2>&1 ; then
    echo "Creating s3://$staging_bucket"
    aws s3 mb "s3://${staging_bucket}"
  fi

  local -r staging_path="${staging_bucket}/devel"

  echo "+++ Staging server tars to S3: s3://${staging_path}"

  # TODO: Region if not us-east?  (s3-us-west-1.amazonaws.com)
  SERVER_BINARY_TAR_URL="s3://${staging_path}/${SERVER_BINARY_TAR##*/}"
  SERVER_BINARY_TAR_DOWNLOAD_URL="http://s3.amazonaws.com/${staging_path}/${SERVER_BINARY_TAR##*/}"

  SALT_TAR_URL="s3://${staging_path}/${SALT_TAR##*/}"
  SALT_TAR_DOWNLOAD_URL="http://s3.amazonaws.com/${staging_path}/${SALT_TAR##*/}"

  mkdir ${KUBE_TEMP}/s3
  cp -p "${SERVER_BINARY_TAR}" ${KUBE_TEMP}/s3/
  cp -p "${SALT_TAR}" ${KUBE_TEMP}/s3/

  # TODO: move to acl private & signed url
  aws s3 sync --acl public-read ${KUBE_TEMP}/s3/ "s3://${staging_path}/"
  #aws s3 cp --acl public-read "${SERVER_BINARY_TAR}" "${SERVER_BINARY_TAR_URL}"
  # TODO: move to acl private & signed url
  #aws s3 cp --acl public-read "${SALT_TAR}" "${SALT_TAR_URL}"
}

# Detect the information about the minions
#
# Assumed vars:
#   MINION_NAMES
#   ZONE
# Vars set:
#   KUBE_MINION_IP_ADDRESS (array)
function detect-minions () {
  KUBE_MINION_IP_ADDRESSES=()
  for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
    # gcutil will print the "external-ip" column header even if no instances are found
    local minion_ip=$(gcutil listinstances --format=csv --sort=external-ip \
      --columns=external-ip --zone ${ZONE} --filter="name eq ${MINION_NAMES[$i]}" \
      | tail -n '+2' | tail -n 1)
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


# Add a tag to the specified resource
#
# Assumed vars:
#   -
# Vars set:
#   -
function set-tag() {
  local resource_id=$1
  local key=$2
  local value=$3
  aws ec2 create-tags --resources ${resource_id} --tags Key=${key},Value=${value}
}

# Add our kubernetes-cluster tag to the specified resource
#
# Assumed vars:
#   CLUSTER_KEY
# Vars set:
#   -
function set-cluster-tag() {
  local resource_id=$1
  set-tag ${resource_id} kubernetes-cluster ${CLUSTER_KEY}
}


# Find the IP for the master
#
# Assumed vars:
#   CLUSTER_KEY
# Vars set:
#   KUBE_MASTER_INSTANCE_ID
#   KUBE_MASTER_PUBLIC_IP
#   KUBE_MASTER_PRIVATE_IP
function find-master () {
  KUBE_MASTER_INSTANCE_ID=$(aws ec2 describe-instances \
                                --filters "Name=tag:kubernetes-cluster,Values=${CLUSTER_KEY}" \
                                          "Name=tag:kubernetes-role,Values=master" \
                                          "Name=instance-state-name,Values=pending,running" \
                                --query Reservations[].Instances[].InstanceId)
  if [[ ! -z "${KUBE_MASTER_INSTANCE_ID-}" ]]; then
    KUBE_MASTER_PUBLIC_IP=$(aws ec2 describe-instances --instance-ids ${KUBE_MASTER_INSTANCE_ID} \
                                --query Reservations[].Instances[].PublicIpAddress)
    KUBE_MASTER_PRIVATE_IP=$(aws ec2 describe-instances --instance-ids ${KUBE_MASTER_INSTANCE_ID} \
                                --query Reservations[].Instances[].PrivateIpAddress)

    echo "Using master: $KUBE_MASTER_PUBLIC_IP / $KUBE_MASTER_PRIVATE_IP"
  fi
}

# Detect the IP for the master
#
# Assumed vars:
#   CLUSTER_KEY
# Vars set:
#   KUBE_MASTER_INSTANCE_ID
#   KUBE_MASTER_PUBLIC_IP
#   KUBE_MASTER_PRIVATE_IP
function detect-master () {
  find-master
  if [[ -z "${KUBE_MASTER_INSTANCE_ID-}" ]]; then
    echo "Could not detect Kubernetes master node.  Make sure you've launched a cluster with 'kube-up.sh'" >&2
    exit 1
  fi
}

# Get or create a VPC
#
# Assumed vars:
#   CLUSTER_KEY
# Vars set:
#   VPC_ID
function ensure-vpc () {
  VPC_ID=`aws ec2 describe-vpcs --filters Name=tag:kubernetes-cluster,Values=${CLUSTER_KEY} --query Vpcs[].VpcId`
  if [[ "${VPC_ID}" == "" ]]; then
    echo "Creating VPC"
    # We aren't allowed to create anything bigger than a /16
    VPC_ID=`aws ec2 create-vpc --cidr 10.100.0.0/16 --query Vpc.VpcId`
    set-cluster-tag ${VPC_ID}
  fi
  echo "Using VPC: ${VPC_ID}"
}

# Get or create a VPC subnet
#
# Assumed vars:
#   CLUSTER_KEY
#   VPC_ID
# Vars set:
#   SUBNET_ID
function ensure-subnet () {
  local AZ=$1
  local NETMASK=$2
  SUBNET_ID=`aws ec2 describe-subnets \
                 --filters Name=tag:kubernetes-cluster,Values=${CLUSTER_KEY} \
                           Name=availability-zone,Values=${AZ} \
                 --query Subnets[0].SubnetId`
  if [[ "${SUBNET_ID}" == "" ]]; then
    echo "Creating subnet in VPC"
    SUBNET_ID=`aws ec2 create-subnet \
                   --vpc-id ${VPC_ID} \
                   --cidr-block ${NETMASK} \
                   --availability-zone=${AZ} \
                   --query Subnet.SubnetId`
    set-cluster-tag ${SUBNET_ID}
  fi
  echo "Using vpc subnet: ${SUBNET_ID}"
}

# Get or create a VPC gateway
#
# Assumed vars:
#   CLUSTER_KEY
#   VPC_ID
# Vars set:
#   GATEWAY_ID
function ensure-gateway () {
  GATEWAY_ID=`aws ec2 describe-internet-gateways \
                  --filters Name=tag:kubernetes-cluster,Values=${CLUSTER_KEY} \
                  --query InternetGateways[0].InternetGatewayId`
  if [[ "${GATEWAY_ID}" == "None" ]]; then
    echo "Creating Internet Gateway in VPC"
    GATEWAY_ID=`aws ec2 create-internet-gateway --query InternetGateway.InternetGatewayId`
    set-cluster-tag ${GATEWAY_ID}
  fi
  echo "Using internet gateway: ${GATEWAY_ID}"

  aws ec2 attach-internet-gateway --internet-gateway-id ${GATEWAY_ID} --vpc-id ${VPC_ID} || echo "Already attached"
}

# Get or create VPC route table
#
# Assumed vars:
#   CLUSTER_KEY
#   VPC_ID
# Vars set:
#   ROUTE_TABLE_ID
function ensure-routetable () {
  ROUTE_TABLE_ID=`aws ec2 describe-route-tables \
                      --filters Name=tag:kubernetes-cluster,Values=${CLUSTER_KEY} \
                                Name=vpc-id,Values=${VPC_ID} \
                      --query RouteTables[].RouteTableId`
  if [[ "${ROUTE_TABLE_ID}" == "" ]]; then
    echo "Creating route table in VPC"
    ROUTE_TABLE_ID=`aws ec2 create-route-table --vpc-id ${VPC_ID} --query RouteTable.RouteTableId`
    set-cluster-tag ${ROUTE_TABLE_ID}
  fi
  echo "Using route table: ${ROUTE_TABLE_ID}"
}

# Get or create VPC route table
#
# Assumed vars:
#   GATEWAY_ID
#   ROUTE_TABLE_ID
# Vars set:
#   -
function ensure-defaultroute () {
  # This is idempotent
  aws ec2 create-route --route-table-id ${ROUTE_TABLE_ID} --destination-cidr-block 0.0.0.0/0 --gateway-id ${GATEWAY_ID} || echo "Route already exists"
}

# Makes sure a security group exists
#
# Assumed vars:
#   GATEWAY_ID
#   ROUTE_TABLE_ID
# Vars set:
#   -
function ensure-securitygroup () {
  local name=$1
  local description=$2

  SECURITY_GROUP_ID=`aws ec2 describe-security-groups --filters Name=vpc-id,Values=${VPC_ID} Name=group-name,Values=${name} --query SecurityGroups[].GroupId`
  if [[ "${SECURITY_GROUP_ID}" == "" ]]; then
    echo "Creating security group ${name}"
    SECURITY_GROUP_ID=`aws ec2 create-security-group --group-name=${name} --vpc-id ${VPC_ID} --description "${description}" --query GroupId`
    set-cluster-tag ${SECURITY_GROUP_ID}
  fi
  echo "Using security group ${SECURITY_GROUP_ID}"
}

# Adds an ingress rule to the current SECURITY_GROUP_ID
#
# Assumed vars:
#   SECURITY_GROUP_ID
# Vars set:
#   -
function authorize-ingress () {
  local protocol=$1
  local cidr=$2
  local from_port=$3
  local to_port=$4

  local ports="${from_port}-${to_port}"

  if [[ ${ports} == '-1--1' ]]; then
    ports="all"
  fi

  FOUND=`aws ec2 describe-security-groups \
             --group-ids ${SECURITY_GROUP_ID} \
             --filters Name=ip-permission.cidr,Values=${cidr} \
                       Name=ip-permission.from-port,Values=${from_port} \
                       Name=ip-permission.to-port,Values=${to_port} \
                       Name=ip-permission.protocol,Values=${protocol} \
             --query SecurityGroups[].GroupId`
  if [[ "${FOUND}" == "" ]]; then
    aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol ${protocol} --cidr ${cidr} --port ${ports}
  fi
}


# Adds an rule for inter-security-group traffic
#
# Assumed vars:
#   -
# Vars set:
#   -
function authorize-internal () {
  local src_sg=$1
  local dest_sg=$2
  local protocol=$3
  local from_port=$4
  local to_port=$5


  local ports="${from_port}-${to_port}"

  if [[ ${ports} == '-1--1' ]]; then
    ports="all"
  fi

  FOUND=`aws ec2 describe-security-groups --group-ids ${dest_sg} --filters Name=ip-permission.group-id,Values=${src_sg} Name=ip-permission.from-port,Values=${from_port} Name=ip-permission.to-port,Values=${to_port} Name=ip-permission.protocol,Values=${protocol} --query SecurityGroups[].GroupId`
  if [[ "${FOUND}" == "" ]]; then
    aws ec2 authorize-security-group-ingress --group-id $dest_sg --source-group ${src_sg} --protocol ${protocol} --port ${ports}
  fi
}

# Chooses an EC2 keypair to use, if we haven't already set KEYPAIR
#
# Vars set:
#   KEYPAIR
function ensure-keypair () {
  if [[ -z "${KEYPAIR-}" ]]; then
    KEYPAIR=$(aws ec2 describe-key-pairs --query KeyPairs[0].KeyName)
    echo "Defaulting to keypair ${KEYPAIR}"
  fi
}

# Chooses an EC2 instance type, if we haven't already set INSTANCETYP
# Actually just chooses m3.medium, for now
#
# Vars set:
#   INSTANCETYPE
function ensure-instancetype () {
  if [[ -z "${INSTANCETYPE-}" ]]; then
    INSTANCETYPE="m3.medium"
    echo "Defaulting to instance type ${INSTANCETYPE}"
  fi
}

# Gets the state name for an EC2 instance
function get-state() {
  local instance_id=$1
  aws ec2 describe-instances --instance-id=${instance_id} --query Reservations[].Instances[].State.Name
}


# Waits for an EC2 instance to be running
function wait-running() {
  local instance_id=$1
  local i=0

  while [[ $i -lt 10 ]]; do
    if [[ $i != 0 ]]; then
      sleep 10
    fi
    state=$(get-state ${instance_id})

    if [[ "${state}" == "running" ]]; then
      echo "Instance is running"
      return 0
    fi

    echo "Instance not running.  Waiting..."
    let i=i+1
  done

  echo "Timed out waiting for instance to start; state was ${state}"
  return 1
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

# Instantiate a kubernetes cluster
#
# Assumed vars
#   KUBE_ROOT
#   <Various vars set in config file>
function kube-up {
  ensure-temp-dir

  # Detect the project into $PROJECT if it isn't set
  detect-project

  # Make sure we have the tar files staged on Google Storage
  find-release-tars
  upload-server-tars

  get-password
  python "${KUBE_ROOT}/third_party/htpasswd/htpasswd.py" \
    -b -c "${KUBE_TEMP}/htpasswd" "$KUBE_USER" "$KUBE_PASSWORD"
  local htpasswd
  htpasswd=$(cat "${KUBE_TEMP}/htpasswd")

  ensure-vpc
  ensure-subnet ${REGION}b 10.100.16.0/20

  ensure-gateway
  ensure-routetable
  ensure-defaultroute

  ensure-securitygroup "kubernetes-default" "Default security group for k8s instances"
  # Allow SSH (to all instances)
  authorize-ingress tcp 0.0.0.0/0 22 22

  # Allow HTTPS (to all instances)
  authorize-ingress tcp 0.0.0.0/0 443 443
  
  # Allow ICMP/pings
  authorize-ingress icmp 0.0.0.0/0 -1 -1

  # Allow instances to talk to each other freely
  authorize-internal ${SECURITY_GROUP_ID} ${SECURITY_GROUP_ID} tcp 1 65535
  authorize-internal ${SECURITY_GROUP_ID} ${SECURITY_GROUP_ID} udp 1 65535

  find-master

  ensure-keypair
  ensure-instancetype

  if [[ -z ${KUBE_MASTER_INSTANCE_ID} ]]; then
    (
      echo "#! /bin/bash"
      echo "mkdir -p /var/cache/kubernetes-install"
      echo "cd /var/cache/kubernetes-install"

      # TODO: Use the internal ec2 dns name?
      echo "readonly MASTER_NAME='127.0.0.1'"
      # TODO: Make this use tags?
      echo "readonly NODE_INSTANCE_PREFIX='${CLUSTER_KEY}-minion'"
      echo "readonly SERVER_BINARY_TAR_URL='${SERVER_BINARY_TAR_DOWNLOAD_URL}'"
      echo "readonly SALT_TAR_URL='${SALT_TAR_DOWNLOAD_URL}'"
      echo "readonly MASTER_HTPASSWD='${htpasswd}'"
      echo "readonly PORTAL_NET='${PORTAL_NET}'"
      echo "readonly FLUENTD_ELASTICSEARCH='${FLUENTD_ELASTICSEARCH:-false}'"
      echo "readonly FLUENTD_GCP='${FLUENTD_GCP:-false}'"
      grep -v "^#" "${KUBE_ROOT}/cluster/ec2/templates/create-dynamic-salt-files.sh"
      grep -v "^#" "${KUBE_ROOT}/cluster/ec2/templates/download-release.sh"
      grep -v "^#" "${KUBE_ROOT}/cluster/ec2/templates/salt-master.sh"
    )  > "${KUBE_TEMP}/master-start.sh"

    INSTANCE_ID=`aws ec2 run-instances --query Instances[].InstanceId \
      --image-id ${AMI} \
      --key-name ${KEYPAIR} \
      --security-group-ids ${SECURITY_GROUP_ID} \
      --instance-type ${INSTANCETYPE} \
      --subnet-id ${SUBNET_ID} \
      --associate-public-ip-address \
      --user-data file://${KUBE_TEMP}/master-start.sh`
    set-cluster-tag ${INSTANCE_ID}
    set-tag ${INSTANCE_ID} kubernetes-id master-0
    set-tag ${INSTANCE_ID} kubernetes-role master
    set-tag ${INSTANCE_ID} Name ${CLUSTER_KEY}-master-0
  fi

  detect-master
  wait-running ${KUBE_MASTER_INSTANCE_ID}

  # Refresh public-ip
  detect-master
  
#  # For logging to GCP we need to enable some minion scopes.
#  if [[ "${FLUENTD_GCP-}" == "true" ]]; then
#     MINION_SCOPES="${MINION_SCOPES}, https://www.googleapis.com/auth/logging.write"
#  fi

  for (( i=0; i<${NUM_MINIONS}; i++)); do
    (
      echo "#! /bin/bash"
      echo "readonly MASTER_NAME='${KUBE_MASTER_PRIVATE_IP}'"
      echo "readonly MINION_ID='${i}'"
#      echo "MINION_IP_RANGE='${MINION_IP_RANGES[$i]}'"
      grep -v "^#" "${KUBE_ROOT}/cluster/ec2/templates/salt-minion.sh"
    ) > "${KUBE_TEMP}/minion-start-${i}.sh"

    INSTANCE_ID=$(aws ec2 describe-instances \
                      --filters "Name=tag:kubernetes-id,Values=minion-${i}" \
                                "Name=tag:kubernetes-cluster,Values=${CLUSTER_KEY}" \
                                "Name=instance-state-name,Values=pending,running" \
                      --query Reservations[].Instances[].InstanceId)
    if [[ ${INSTANCE_ID} == "" ]]; then
      INSTANCE_ID=`aws ec2 run-instances --query Instances[].InstanceId \
        --image-id ${AMI} \
        --key-name ${KEYPAIR} \
        --security-group-ids ${SECURITY_GROUP_ID} \
        --instance-type ${INSTANCETYPE} \
        --subnet-id ${SUBNET_ID} \
        --associate-public-ip-address \
        --user-data file://${KUBE_TEMP}/minion-start-${i}.sh`
      set-cluster-tag ${INSTANCE_ID}
      set-tag ${INSTANCE_ID} kubernetes-id minion-${i}
      set-tag ${INSTANCE_ID} kubernetes-role minion
      set-tag ${INSTANCE_ID} Name ${CLUSTER_KEY}-minion-${i}
    fi

    wait-running ${INSTANCE_ID}
  done

  echo "Waiting for cluster initialization."
  echo
  echo "  This will continually check to see if the API for kubernetes is reachable."
  echo "  This might loop forever if there was some uncaught error during start"
  echo "  up."
  echo

  until curl --insecure --user "${KUBE_USER}:${KUBE_PASSWORD}" --max-time 5 \
          --fail --output /dev/null --silent "https://${KUBE_MASTER_PUBLIC_IP}/api/v1beta1/pods"; do
      printf "."
      sleep 2
  done

  echo "Kubernetes cluster created."
  echo "Sanity checking cluster..."

  sleep 5

  ## Basic sanity checking
  #local i
  #local rc # Capture return code without exiting because of errexit bash option
  #for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
  #    # Make sure docker is installed
  #    gcutil ssh "${MINION_NAMES[$i]}" which docker >/dev/null || {
  #      echo "Docker failed to install on ${MINION_NAMES[$i]}. Your cluster is unlikely" >&2
  #      echo "to work correctly. Please run ./cluster/kube-down.sh and re-create the" >&2
  #      echo "cluster. (sorry!)" >&2
  #      exit 1
  #    }
  #done

  echo
  echo "Kubernetes cluster is running.  The master is running at:"
  echo
  echo "  https://${KUBE_MASTER_PUBLIC_IP}"
  echo
  echo "The user name and password to use is located in ~/.kubernetes_auth."
  echo

  local kube_cert=".kubecfg.crt"
  local kube_key=".kubecfg.key"
  local ca_cert=".kubernetes.ca.crt"

  # TODO: generate ADMIN (and KUBELET) tokens and put those in the master's
  # config file.  Distribute the same way the htpasswd is done.
  (umask 077
   ssh ${SSH_USER}@${KUBE_MASTER_PUBLIC_IP} sudo cat /usr/share/nginx/kubecfg.crt >"${HOME}/${kube_cert}" 2>/dev/null
   ssh ${SSH_USER}@${KUBE_MASTER_PUBLIC_IP} sudo cat /usr/share/nginx/kubecfg.key >"${HOME}/${kube_key}" 2>/dev/null
   ssh ${SSH_USER}@${KUBE_MASTER_PUBLIC_IP} sudo cat /usr/share/nginx/ca.crt >"${HOME}/${ca_cert}" 2>/dev/null

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
  )
}

# Delete a kubernetes cluster
function kube-down {
  # Detect the project into $PROJECT
  detect-project

  echo "Bringing down cluster"
  gcutil deletefirewall  \
    --project "${PROJECT}" \
    --norespect_terminal_width \
    --sleep_between_polls "${POLL_SLEEP_INTERVAL}" \
    --force \
    "${MASTER_NAME}-https" &

  gcutil deleteinstance \
    --project "${PROJECT}" \
    --norespect_terminal_width \
    --sleep_between_polls "${POLL_SLEEP_INTERVAL}" \
    --force \
    --delete_boot_pd \
    --zone "${ZONE}" \
    "${MASTER_NAME}" &

  gcutil deletefirewall  \
    --project "${PROJECT}" \
    --norespect_terminal_width \
    --sleep_between_polls "${POLL_SLEEP_INTERVAL}" \
    --force \
    "${MINION_NAMES[@]/%/-all}" &

  gcutil deleteinstance \
    --project "${PROJECT}" \
    --norespect_terminal_width \
    --sleep_between_polls "${POLL_SLEEP_INTERVAL}" \
    --force \
    --delete_boot_pd \
    --zone "${ZONE}" \
    "${MINION_NAMES[@]}" &

  gcutil deleteroute  \
    --project "${PROJECT}" \
    --norespect_terminal_width \
    --sleep_between_polls "${POLL_SLEEP_INTERVAL}" \
    --force \
    "${MINION_NAMES[@]}" &

  wait

}

# Update a kubernetes cluster with latest source
function kube-push {
  detect-project
  detect-master

  # Make sure we have the tar files staged on Google Storage
  ensure-temp-dir
  find-release-tars
  upload-server-tars

  (
    echo "#! /bin/bash"
    echo "mkdir -p /var/cache/kubernetes-install"
    echo "cd /var/cache/kubernetes-install"
    echo "readonly SERVER_BINARY_TAR_URL='${SERVER_BINARY_TAR_DOWNLOAD_URL}'"
    echo "readonly SALT_TAR_URL='${SALT_TAR_DOWNLOAD_URL}'"
    grep -v "^#" "${KUBE_ROOT}/cluster/ec2/templates/download-release.sh"
    echo "echo Executing configuration"
    echo "sudo salt '*' mine.update"
    echo "sudo salt --force-color '*' state.highstate"
  ) | ssh ${SSH_USER}@${KUBE_MASTER_PUBLIC_IP} sudo bash

  get-password

  echo
  echo "Kubernetes cluster is running.  The master is running at:"
  echo
  echo "  https://${KUBE_MASTER_PUBLIC_IP}"
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
  # ec2 specific
  detect-project

  # Open up port 80 & 8080 so common containers on minions can be reached
  gcutil addfirewall \
    --project "${PROJECT}" \
    --norespect_terminal_width \
    --sleep_between_polls "${POLL_SLEEP_INTERVAL}" \
    --target_tags "${MINION_TAG}" \
    --allowed tcp:80,tcp:8080 \
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
  gcutil deletefirewall  \
    --project "${PROJECT}" \
    --norespect_terminal_width \
    --sleep_between_polls "${POLL_SLEEP_INTERVAL}" \
    --force \
    "${MINION_TAG}-${INSTANCE_PREFIX}-http-alt" || true > /dev/null
  "${KUBE_ROOT}/cluster/kube-down.sh" > /dev/null
}

# SSH to a node by name ($1) and run a command ($2).
function ssh-to-node {
  local node="$1"
  local cmd="$2"
  gcutil --log_level=WARNING ssh --ssh_arg "-o LogLevel=quiet" "${node}" "${cmd}"
}

# Restart the kube-proxy on a node ($1)
function restart-kube-proxy {
  ssh-to-node "$1" "sudo /etc/init.d/kube-proxy restart"
}
