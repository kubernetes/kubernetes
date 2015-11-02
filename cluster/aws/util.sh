#!/bin/bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
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
source "${KUBE_ROOT}/cluster/aws/${KUBE_CONFIG_FILE-"config-default.sh"}"
source "${KUBE_ROOT}/cluster/common.sh"

ALLOCATE_NODE_CIDRS=true

NODE_INSTANCE_PREFIX="${INSTANCE_PREFIX}-minion"
ASG_NAME="${NODE_INSTANCE_PREFIX}-group"

# We could allow the master disk volume id to be specified in future
MASTER_DISK_ID=

# Defaults: ubuntu -> vivid
if [[ "${KUBE_OS_DISTRIBUTION}" == "ubuntu" ]]; then
  KUBE_OS_DISTRIBUTION=vivid
fi

case "${KUBE_OS_DISTRIBUTION}" in
  trusty|wheezy|jessie|vivid|coreos)
    source "${KUBE_ROOT}/cluster/aws/${KUBE_OS_DISTRIBUTION}/util.sh"
    ;;
  *)
    echo "Cannot start cluster using os distro: ${KUBE_OS_DISTRIBUTION}" >&2
    exit 2
    ;;
esac

# This removes the final character in bash (somehow)
AWS_REGION=${ZONE%?}

export AWS_DEFAULT_REGION=${AWS_REGION}
AWS_CMD="aws --output json ec2"
AWS_ELB_CMD="aws --output json elb"
AWS_ASG_CMD="aws --output json autoscaling"

INTERNAL_IP_BASE=172.20.0
MASTER_IP_SUFFIX=.9
MASTER_INTERNAL_IP=${INTERNAL_IP_BASE}${MASTER_IP_SUFFIX}

MASTER_SG_NAME="kubernetes-master-${CLUSTER_ID}"
MINION_SG_NAME="kubernetes-minion-${CLUSTER_ID}"

# Be sure to map all the ephemeral drives.  We can specify more than we actually have.
# TODO: Actually mount the correct number (especially if we have more), though this is non-trivial, and
#  only affects the big storage instance types, which aren't a typical use case right now.
BLOCK_DEVICE_MAPPINGS_BASE="{\"DeviceName\": \"/dev/sdc\",\"VirtualName\":\"ephemeral0\"},{\"DeviceName\": \"/dev/sdd\",\"VirtualName\":\"ephemeral1\"},{\"DeviceName\": \"/dev/sde\",\"VirtualName\":\"ephemeral2\"},{\"DeviceName\": \"/dev/sdf\",\"VirtualName\":\"ephemeral3\"}"
MASTER_BLOCK_DEVICE_MAPPINGS="[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"DeleteOnTermination\":true,\"VolumeSize\":${MASTER_ROOT_DISK_SIZE},\"VolumeType\":\"${MASTER_ROOT_DISK_TYPE}\"}}, ${BLOCK_DEVICE_MAPPINGS_BASE}]"
MINION_BLOCK_DEVICE_MAPPINGS="[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"DeleteOnTermination\":true,\"VolumeSize\":${MINION_ROOT_DISK_SIZE},\"VolumeType\":\"${MINION_ROOT_DISK_TYPE}\"}}, ${BLOCK_DEVICE_MAPPINGS_BASE}]"

function json_val {
    python -c 'import json,sys;obj=json.load(sys.stdin);print obj'$1''
}

# TODO (ayurchuk) Refactor the get_* functions to use filters
# TODO (bburns) Parameterize this for multiple cluster per project

function get_vpc_id {
  $AWS_CMD --output text describe-vpcs \
           --filters Name=tag:Name,Values=kubernetes-vpc \
                     Name=tag:KubernetesCluster,Values=${CLUSTER_ID} \
           --query Vpcs[].VpcId
}

function get_subnet_id {
  python -c "import json,sys; lst = [str(subnet['SubnetId']) for subnet in json.load(sys.stdin)['Subnets'] if subnet['VpcId'] == '$1' and subnet['AvailabilityZone'] == '$2']; print ''.join(lst)"
}

function get_igw_id {
  python -c "import json,sys; lst = [str(igw['InternetGatewayId']) for igw in json.load(sys.stdin)['InternetGateways'] for attachment in igw['Attachments'] if attachment['VpcId'] == '$1']; print ''.join(lst)"
}

function get_route_table_id {
  python -c "import json,sys; lst = [str(route_table['RouteTableId']) for route_table in json.load(sys.stdin)['RouteTables'] if route_table['VpcId'] == '$1']; print ''.join(lst)"
}

function get_elbs_in_vpc {
 # ELB doesn't seem to be on the same platform as the rest of AWS; doesn't support filtering
  $AWS_ELB_CMD describe-load-balancers | \
    python -c "import json,sys; lst = [str(lb['LoadBalancerName']) for lb in json.load(sys.stdin)['LoadBalancerDescriptions'] if lb['VPCId'] == '$1']; print '\n'.join(lst)"
}

function expect_instance_states {
  python -c "import json,sys; lst = [str(instance['InstanceId']) for reservation in json.load(sys.stdin)['Reservations'] for instance in reservation['Instances'] if instance['State']['Name'] != '$1']; print ' '.join(lst)"
}

function get_instanceid_from_name {
  local tagName=$1
  $AWS_CMD --output text describe-instances \
    --filters Name=tag:Name,Values=${tagName} \
              Name=instance-state-name,Values=running \
              Name=tag:KubernetesCluster,Values=${CLUSTER_ID} \
    --query Reservations[].Instances[].InstanceId
}

function get_instance_public_ip {
  local instance_id=$1
  $AWS_CMD --output text describe-instances \
    --instance-ids ${instance_id} \
    --query Reservations[].Instances[].NetworkInterfaces[0].Association.PublicIp
}

function get_instance_private_ip {
  local instance_id=$1
  $AWS_CMD --output text describe-instances \
    --instance-ids ${instance_id} \
    --query Reservations[].Instances[].NetworkInterfaces[0].PrivateIpAddress
}

# Gets a security group id, by name ($1)
function get_security_group_id {
  local name=$1
  $AWS_CMD --output text describe-security-groups \
           --filters Name=vpc-id,Values=${VPC_ID} \
                     Name=group-name,Values=${name} \
                     Name=tag:KubernetesCluster,Values=${CLUSTER_ID} \
           --query SecurityGroups[].GroupId \
  | tr "\t" "\n"
}

function detect-master () {
  KUBE_MASTER=${MASTER_NAME}
  if [[ -z "${KUBE_MASTER_ID-}" ]]; then
    KUBE_MASTER_ID=$(get_instanceid_from_name ${MASTER_NAME})
  fi
  if [[ -z "${KUBE_MASTER_ID-}" ]]; then
    echo "Could not detect Kubernetes master node.  Make sure you've launched a cluster with 'kube-up.sh'"
    exit 1
  fi
  if [[ -z "${KUBE_MASTER_IP-}" ]]; then
    KUBE_MASTER_IP=$(get_instance_public_ip ${KUBE_MASTER_ID})
  fi
  if [[ -z "${KUBE_MASTER_IP-}" ]]; then
    echo "Could not detect Kubernetes master node IP.  Make sure you've launched a cluster with 'kube-up.sh'"
    exit 1
  fi
  echo "Using master: $KUBE_MASTER (external IP: $KUBE_MASTER_IP)"
}


function query-running-minions () {
  local query=$1
  $AWS_CMD --output text describe-instances \
           --filters Name=instance-state-name,Values=running \
                     Name=vpc-id,Values=${VPC_ID} \
                     Name=tag:KubernetesCluster,Values=${CLUSTER_ID} \
                     Name=tag:Role,Values=${MINION_TAG} \
           --query ${query}
}

function find-running-minions () {
  MINION_IDS=()
  MINION_NAMES=()
  for id in $(query-running-minions "Reservations[].Instances[].InstanceId"); do
    MINION_IDS+=("${id}")

    # We use the minion ids as the name
    MINION_NAMES+=("${id}")
  done
}

function detect-minions () {
  find-running-minions

  # This is inefficient, but we want MINION_NAMES / MINION_IDS to be ordered the same as KUBE_MINION_IP_ADDRESSES
  KUBE_MINION_IP_ADDRESSES=()
  for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
    local minion_ip
    if [[ "${ENABLE_MINION_PUBLIC_IP}" == "true" ]]; then
      minion_ip=$(get_instance_public_ip ${MINION_NAMES[$i]})
    else
      minion_ip=$(get_instance_private_ip ${MINION_NAMES[$i]})
    fi
    echo "Found minion ${i}: ${MINION_NAMES[$i]} @ ${minion_ip}"
    KUBE_MINION_IP_ADDRESSES+=("${minion_ip}")
  done

  if [[ -z "$KUBE_MINION_IP_ADDRESSES" ]]; then
    echo "Could not detect Kubernetes minion nodes.  Make sure you've launched a cluster with 'kube-up.sh'"
    exit 1
  fi
}

function detect-security-groups {
  if [[ -z "${MASTER_SG_ID-}" ]]; then
    MASTER_SG_ID=$(get_security_group_id "${MASTER_SG_NAME}")
    if [[ -z "${MASTER_SG_ID}" ]]; then
      echo "Could not detect Kubernetes master security group.  Make sure you've launched a cluster with 'kube-up.sh'"
      exit 1
    else
      echo "Using master security group: ${MASTER_SG_NAME} ${MASTER_SG_ID}"
    fi
  fi
  if [[ -z "${MINION_SG_ID-}" ]]; then
    MINION_SG_ID=$(get_security_group_id "${MINION_SG_NAME}")
    if [[ -z "${MINION_SG_ID}" ]]; then
      echo "Could not detect Kubernetes minion security group.  Make sure you've launched a cluster with 'kube-up.sh'"
      exit 1
    else
      echo "Using minion security group: ${MINION_SG_NAME} ${MINION_SG_ID}"
    fi
  fi
}

# Detects the AMI to use (considering the region)
# This really should be in the various distro-specific util functions,
# but CoreOS uses this for the master, so for now it is here.
#
# TODO: Remove this and just have each distro implement detect-image
#
# Vars set:
#   AWS_IMAGE
function detect-image () {
case "${KUBE_OS_DISTRIBUTION}" in
  trusty|coreos)
    detect-trusty-image
    ;;
  vivid)
    detect-vivid-image
    ;;
  wheezy)
    detect-wheezy-image
    ;;
  jessie)
    detect-jessie-image
    ;;
  *)
    echo "Please specify AWS_IMAGE directly (distro not recognized)"
    exit 2
    ;;
esac
}

# Detects the AMI to use for trusty (considering the region)
# Used by CoreOS & Ubuntu
#
# Vars set:
#   AWS_IMAGE
function detect-trusty-image () {
  # This is the ubuntu 14.04 image for <region>, amd64, hvm:ebs-ssd
  # See here: http://cloud-images.ubuntu.com/locator/ec2/ for other images
  # This will need to be updated from time to time as amis are deprecated
  if [[ -z "${AWS_IMAGE-}" ]]; then
    case "${AWS_REGION}" in
      ap-northeast-1)
        AWS_IMAGE=ami-93876e93
        ;;

      ap-southeast-1)
        AWS_IMAGE=ami-66546234
        ;;

      eu-central-1)
        AWS_IMAGE=ami-e2a694ff
        ;;

      eu-west-1)
        AWS_IMAGE=ami-d7fd6ea0
        ;;

      sa-east-1)
        AWS_IMAGE=ami-a357eebe
        ;;

      us-east-1)
        AWS_IMAGE=ami-6089d208
        ;;

      us-west-1)
        AWS_IMAGE=ami-cf7d998b
        ;;

      cn-north-1)
        AWS_IMAGE=ami-d436a4ed
        ;;

      us-gov-west-1)
        AWS_IMAGE=ami-01523322
        ;;

      ap-southeast-2)
        AWS_IMAGE=ami-cd4e3ff7
        ;;

      us-west-2)
        AWS_IMAGE=ami-3b14370b
        ;;

      *)
        echo "Please specify AWS_IMAGE directly (region not recognized)"
        exit 1
    esac
  fi
}

# Computes the AWS fingerprint for a public key file ($1)
# $1: path to public key file
# Note that this is a different hash from the OpenSSH hash.
# But AWS gives us this public key hash in the describe keys output, so we should stick with this format.
# Hopefully this will be done by the aws cli tool one day: https://github.com/aws/aws-cli/issues/191
# NOTE: This does not work on Mavericks, due to an odd ssh-keygen version, so we use get-ssh-fingerprint instead
function get-aws-fingerprint {
  local -r pubkey_path=$1
  ssh-keygen -f ${pubkey_path} -e -m PKCS8  | openssl rsa -pubin -outform DER | openssl md5 -c | sed -e 's/(stdin)= //g'
}

# Computes the SSH fingerprint for a public key file ($1)
# #1: path to public key file
# Note this is different from the AWS fingerprint; see notes on get-aws-fingerprint
function get-ssh-fingerprint {
  local -r pubkey_path=$1
  ssh-keygen -lf ${pubkey_path} | cut -f2 -d' '
}

# Import an SSH public key to AWS.
# Ignores duplicate names; recommended to use a name that includes the public key hash.
# $1 name
# $2 public key path
function import-public-key {
  local -r name=$1
  local -r path=$2

  local ok=1
  local output=""
  output=$($AWS_CMD import-key-pair --key-name ${name} --public-key-material "file://${path}" 2>&1) || ok=0
  if [[ ${ok} == 0 ]]; then
    # Idempotency: ignore if duplicate name
    if [[ "${output}" != *"InvalidKeyPair.Duplicate"* ]]; then
      echo "Error importing public key"
      echo "Output: ${output}"
      exit 1
    fi
  fi
}

# Robustly try to create a security group, if it does not exist.
# $1: The name of security group; will be created if not exists
# $2: Description for security group (used if created)
#
# Note that this doesn't actually return the sgid; we need to re-query
function create-security-group {
  local -r name=$1
  local -r description=$2

  local sgid=$(get_security_group_id "${name}")
  if [[ -z "$sgid" ]]; then
	  echo "Creating security group ${name}."
	  sgid=$($AWS_CMD create-security-group --group-name "${name}" --description "${description}" --vpc-id "${VPC_ID}" --query GroupId --output text)
	  add-tag $sgid KubernetesCluster ${CLUSTER_ID}
  fi
}

# Authorize ingress to a security group.
# Attempts to be idempotent, though we end up checking the output looking for error-strings.
# $1 group-id
# $2.. arguments to pass to authorize-security-group-ingress
function authorize-security-group-ingress {
  local -r sgid=$1
  shift
  local ok=1
  local output=""
  output=$($AWS_CMD authorize-security-group-ingress --group-id "${sgid}" $@ 2>&1) || ok=0
  if [[ ${ok} == 0 ]]; then
    # Idempotency: ignore if duplicate rule
    if [[ "${output}" != *"InvalidPermission.Duplicate"* ]]; then
      echo "Error creating security group ingress rule"
      echo "Output: ${output}"
      exit 1
    fi
  fi
}

# Gets master persistent volume, if exists
# Sets MASTER_DISK_ID
function find-master-pd {
  local name=${MASTER_NAME}-pd
  if [[ -z "${MASTER_DISK_ID}" ]]; then
    MASTER_DISK_ID=`$AWS_CMD --output text describe-volumes \
                             --filters Name=availability-zone,Values=${ZONE} \
                                       Name=tag:Name,Values=${name} \
                                       Name=tag:KubernetesCluster,Values=${CLUSTER_ID} \
                             --query Volumes[].VolumeId`
  fi
}

# Gets or creates master persistent volume
# Sets MASTER_DISK_ID
function ensure-master-pd {
  local name=${MASTER_NAME}-pd

  find-master-pd

  if [[ -z "${MASTER_DISK_ID}" ]]; then
    echo "Creating master disk: size ${MASTER_DISK_SIZE}GB, type ${MASTER_DISK_TYPE}"
    MASTER_DISK_ID=`$AWS_CMD create-volume --availability-zone ${ZONE} --volume-type ${MASTER_DISK_TYPE} --size ${MASTER_DISK_SIZE} --query VolumeId --output text`
    add-tag ${MASTER_DISK_ID} Name ${name}
    add-tag ${MASTER_DISK_ID} KubernetesCluster ${CLUSTER_ID}
  fi
}

# Verify prereqs
function verify-prereqs {
  if [[ "$(which aws)" == "" ]]; then
    echo "Can't find aws in PATH, please fix and retry."
    exit 1
  fi
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

# Take the local tar files and upload them to S3.  They will then be
# downloaded by the master as part of the start up script for the master.
#
# Assumed vars:
#   SERVER_BINARY_TAR
#   SALT_TAR
# Vars set:
#   SERVER_BINARY_TAR_URL
#   SALT_TAR_URL
function upload-server-tars() {
  SERVER_BINARY_TAR_URL=
  SALT_TAR_URL=

  ensure-temp-dir

  if [[ -z ${AWS_S3_BUCKET-} ]]; then
      local project_hash=
      local key=$(aws configure get aws_access_key_id)
      if which md5 > /dev/null 2>&1; then
        project_hash=$(md5 -q -s "${USER} ${key}")
      else
        project_hash=$(echo -n "${USER} ${key}" | md5sum | awk '{ print $1 }')
      fi
      AWS_S3_BUCKET="kubernetes-staging-${project_hash}"
  fi

  echo "Uploading to Amazon S3"

  if ! aws s3api get-bucket-location --bucket ${AWS_S3_BUCKET} > /dev/null 2>&1 ; then
    echo "Creating ${AWS_S3_BUCKET}"

    # Buckets must be globally uniquely named, so always create in a known region
    # We default to us-east-1 because that's the canonical region for S3,
    # and then the bucket is most-simply named (s3.amazonaws.com)
    aws s3 mb "s3://${AWS_S3_BUCKET}" --region ${AWS_S3_REGION}

    local attempt=0
    while true; do
      if ! aws s3 ls --region ${AWS_S3_REGION} "s3://${AWS_S3_BUCKET}" > /dev/null 2>&1; then
        if (( attempt > 5 )); then
          echo
          echo -e "${color_red}Unable to confirm bucket creation." >&2
          echo "Please ensure that s3://${AWS_S3_BUCKET} exists" >&2
          echo -e "and run the script again. (sorry!)${color_norm}" >&2
          exit 1
        fi
      else
        break
      fi
      attempt=$(($attempt+1))
      sleep 1
    done
  fi

  local s3_bucket_location=$(aws --output text s3api get-bucket-location --bucket ${AWS_S3_BUCKET})
  local s3_url_base=https://s3-${s3_bucket_location}.amazonaws.com
  if [[ "${s3_bucket_location}" == "None" ]]; then
    # "US Classic" does not follow the pattern
    s3_url_base=https://s3.amazonaws.com
    s3_bucket_location=us-east-1
  fi

  local -r staging_path="devel"

  local -r local_dir="${KUBE_TEMP}/s3/"
  mkdir ${local_dir}

  echo "+++ Staging server tars to S3 Storage: ${AWS_S3_BUCKET}/${staging_path}"
  local server_binary_path="${staging_path}/${SERVER_BINARY_TAR##*/}"
  cp -a "${SERVER_BINARY_TAR}" ${local_dir}
  cp -a "${SALT_TAR}" ${local_dir}

  aws s3 sync --region ${s3_bucket_location} --exact-timestamps ${local_dir} "s3://${AWS_S3_BUCKET}/${staging_path}/"

  aws s3api put-object-acl --region ${s3_bucket_location} --bucket ${AWS_S3_BUCKET} --key "${server_binary_path}" --grant-read 'uri="http://acs.amazonaws.com/groups/global/AllUsers"'
  SERVER_BINARY_TAR_URL="${s3_url_base}/${AWS_S3_BUCKET}/${server_binary_path}"

  local salt_tar_path="${staging_path}/${SALT_TAR##*/}"
  aws s3api put-object-acl --region ${s3_bucket_location} --bucket ${AWS_S3_BUCKET} --key "${salt_tar_path}" --grant-read 'uri="http://acs.amazonaws.com/groups/global/AllUsers"'
  SALT_TAR_URL="${s3_url_base}/${AWS_S3_BUCKET}/${salt_tar_path}"
}

# Adds a tag to an AWS resource
# usage: add-tag <resource-id> <tag-name> <tag-value>
function add-tag {
  echo "Adding tag to ${1}: ${2}=${3}"

  # We need to retry in case the resource isn't yet fully created
  n=0
  until [ $n -ge 25 ]; do
    $AWS_CMD create-tags --resources ${1} --tags Key=${2},Value=${3} > $LOG && return
    n=$[$n+1]
    sleep 3
  done

  echo "Unable to add tag to AWS resource"
  exit 1
}

# Creates the IAM profile, based on configuration files in templates/iam
function create-iam-profile {
  local key=$1

  local conf_dir=file://${KUBE_ROOT}/cluster/aws/templates/iam

  echo "Creating IAM role: ${key}"
  aws iam create-role --role-name ${key} --assume-role-policy-document ${conf_dir}/${key}-role.json > $LOG

  echo "Creating IAM role-policy: ${key}"
  aws iam put-role-policy --role-name ${key} --policy-name ${key} --policy-document ${conf_dir}/${key}-policy.json > $LOG

  echo "Creating IAM instance-policy: ${key}"
  aws iam create-instance-profile --instance-profile-name ${key} > $LOG

  echo "Adding IAM role to instance-policy: ${key}"
  aws iam add-role-to-instance-profile --instance-profile-name ${key} --role-name ${key} > $LOG
}

# Creates the IAM roles (if they do not already exist)
function ensure-iam-profiles {
  aws iam get-instance-profile --instance-profile-name ${IAM_PROFILE_MASTER} || {
    echo "Creating master IAM profile: ${IAM_PROFILE_MASTER}"
    create-iam-profile ${IAM_PROFILE_MASTER}
  }
  aws iam get-instance-profile --instance-profile-name ${IAM_PROFILE_MINION} || {
    echo "Creating minion IAM profile: ${IAM_PROFILE_MINION}"
    create-iam-profile ${IAM_PROFILE_MINION}
  }
}

# Wait for instance to be in running state
function wait-for-instance-running {
  instance_id=$1
  while true; do
    instance_state=$($AWS_CMD describe-instances --instance-ids ${instance_id} | expect_instance_states running)
    if [[ "$instance_state" == "" ]]; then
      break
    else
      echo "Waiting for instance ${instance_id} to spawn"
      echo "Sleeping for 3 seconds..."
      sleep 3
    fi
  done
}

# Allocates new Elastic IP from Amazon
# Output: allocated IP address
function allocate-elastic-ip {
  $AWS_CMD allocate-address --domain vpc --output text | cut -f3
}

function assign-ip-to-instance {
  local ip_address=$1
  local instance_id=$2
  local fallback_ip=$3

  local elastic_ip_allocation_id=$($AWS_CMD describe-addresses --public-ips $ip_address --output text | cut -f2)
  local association_result=$($AWS_CMD associate-address --instance-id ${master_instance_id} --allocation-id ${elastic_ip_allocation_id} > /dev/null && echo "success" || echo "failure")

  if [[ $association_result = "success" ]]; then
    echo "${ip_address}"
  else
    echo "${fallback_ip}"
  fi
}

# If MASTER_RESERVED_IP looks like IP address, will try to assign it to master instance
# If MASTER_RESERVED_IP is "auto", will allocate new elastic ip and assign that
# If none of the above or something fails, will output originally assigne IP
# Output: assigned IP address
function assign-elastic-ip {
  local assigned_public_ip=$1
  local master_instance_id=$2

  # Check that MASTER_RESERVED_IP looks like an IPv4 address
  if [[ "${MASTER_RESERVED_IP}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    assign-ip-to-instance "${MASTER_RESERVED_IP}" "${master_instance_id}" "${assigned_public_ip}"
  elif [[ "${MASTER_RESERVED_IP}" = "auto" ]]; then
    assign-ip-to-instance $(allocate-elastic-ip) "${master_instance_id}" "${assigned_public_ip}"
  else
    echo "${assigned_public_ip}"
  fi
}


function kube-up {
  echo "Starting cluster using os distro: ${KUBE_OS_DISTRIBUTION}" >&2

  get-tokens

  detect-image
  detect-minion-image

  find-release-tars

  ensure-temp-dir

  upload-server-tars

  ensure-iam-profiles

  load-or-gen-kube-basicauth

  if [[ ! -f "$AWS_SSH_KEY" ]]; then
    ssh-keygen -f "$AWS_SSH_KEY" -N ''
  fi

  # Note that we use get-ssh-fingerprint, so this works on OSX Mavericks
  # get-aws-fingerprint gives the same fingerprint that AWS computes,
  # but OSX Mavericks ssh-keygen can't compute it
  AWS_SSH_KEY_FINGERPRINT=$(get-ssh-fingerprint ${AWS_SSH_KEY}.pub)
  echo "Using SSH key with (AWS) fingerprint: ${AWS_SSH_KEY_FINGERPRINT}"
  AWS_SSH_KEY_NAME="kubernetes-${AWS_SSH_KEY_FINGERPRINT//:/}"

  import-public-key ${AWS_SSH_KEY_NAME} ${AWS_SSH_KEY}.pub

  if [[ -z "${VPC_ID:-}" ]]; then
    VPC_ID=$(get_vpc_id)
  fi
  if [[ -z "$VPC_ID" ]]; then
	  echo "Creating vpc."
	  VPC_ID=$($AWS_CMD create-vpc --cidr-block $INTERNAL_IP_BASE.0/16 | json_val '["Vpc"]["VpcId"]')
	  $AWS_CMD modify-vpc-attribute --vpc-id $VPC_ID --enable-dns-support '{"Value": true}' > $LOG
	  $AWS_CMD modify-vpc-attribute --vpc-id $VPC_ID --enable-dns-hostnames '{"Value": true}' > $LOG
	  add-tag $VPC_ID Name kubernetes-vpc
	  add-tag $VPC_ID KubernetesCluster ${CLUSTER_ID}
  fi

  echo "Using VPC $VPC_ID"

  if [[ -z "${SUBNET_ID:-}" ]]; then
    SUBNET_ID=$($AWS_CMD describe-subnets --filters Name=tag:KubernetesCluster,Values=${CLUSTER_ID} | get_subnet_id $VPC_ID $ZONE)
  fi
  if [[ -z "$SUBNET_ID" ]]; then
    echo "Creating subnet."
    SUBNET_ID=$($AWS_CMD create-subnet --cidr-block $INTERNAL_IP_BASE.0/24 --vpc-id $VPC_ID --availability-zone ${ZONE} | json_val '["Subnet"]["SubnetId"]')
    add-tag $SUBNET_ID KubernetesCluster ${CLUSTER_ID}
  else
    EXISTING_CIDR=$($AWS_CMD describe-subnets --subnet-ids ${SUBNET_ID} --query Subnets[].CidrBlock --output text)
    echo "Using existing CIDR $EXISTING_CIDR"
    INTERNAL_IP_BASE=${EXISTING_CIDR%.*}
    MASTER_INTERNAL_IP=${INTERNAL_IP_BASE}${MASTER_IP_SUFFIX}
  fi

  echo "Using subnet $SUBNET_ID"

  IGW_ID=$($AWS_CMD describe-internet-gateways | get_igw_id $VPC_ID)
  if [[ -z "$IGW_ID" ]]; then
	  echo "Creating Internet Gateway."
	  IGW_ID=$($AWS_CMD create-internet-gateway | json_val '["InternetGateway"]["InternetGatewayId"]')
	  $AWS_CMD attach-internet-gateway --internet-gateway-id $IGW_ID --vpc-id $VPC_ID > $LOG
  fi

  echo "Using Internet Gateway $IGW_ID"

  echo "Associating route table."
  ROUTE_TABLE_ID=$($AWS_CMD --output text describe-route-tables \
                            --filters Name=vpc-id,Values=${VPC_ID} \
                                      Name=tag:KubernetesCluster,Values=${CLUSTER_ID} \
                            --query RouteTables[].RouteTableId)
  if [[ -z "${ROUTE_TABLE_ID}" ]]; then
    echo "Creating route table"
    ROUTE_TABLE_ID=$($AWS_CMD --output text create-route-table \
                              --vpc-id=${VPC_ID} \
                              --query RouteTable.RouteTableId)
    add-tag ${ROUTE_TABLE_ID} KubernetesCluster ${CLUSTER_ID}
  fi

  echo "Associating route table ${ROUTE_TABLE_ID} to subnet ${SUBNET_ID}"
  $AWS_CMD associate-route-table --route-table-id $ROUTE_TABLE_ID --subnet-id $SUBNET_ID > $LOG || true
  echo "Adding route to route table ${ROUTE_TABLE_ID}"
  $AWS_CMD create-route --route-table-id $ROUTE_TABLE_ID --destination-cidr-block 0.0.0.0/0 --gateway-id $IGW_ID > $LOG || true

  echo "Using Route Table $ROUTE_TABLE_ID"

  # Create security groups
  MASTER_SG_ID=$(get_security_group_id "${MASTER_SG_NAME}")
  if [[ -z "${MASTER_SG_ID}" ]]; then
    echo "Creating master security group."
    create-security-group "${MASTER_SG_NAME}" "Kubernetes security group applied to master nodes"
  fi
  MINION_SG_ID=$(get_security_group_id "${MINION_SG_NAME}")
  if [[ -z "${MINION_SG_ID}" ]]; then
    echo "Creating minion security group."
    create-security-group "${MINION_SG_NAME}" "Kubernetes security group applied to minion nodes"
  fi

  detect-security-groups

  # Masters can talk to master
  authorize-security-group-ingress "${MASTER_SG_ID}" "--source-group ${MASTER_SG_ID} --protocol all"

  # Minions can talk to minions
  authorize-security-group-ingress "${MINION_SG_ID}" "--source-group ${MINION_SG_ID} --protocol all"

  # Masters and minions can talk to each other
  authorize-security-group-ingress "${MASTER_SG_ID}" "--source-group ${MINION_SG_ID} --protocol all"
  authorize-security-group-ingress "${MINION_SG_ID}" "--source-group ${MASTER_SG_ID} --protocol all"

  # TODO(justinsb): Would be fairly easy to replace 0.0.0.0/0 in these rules

  # SSH is open to the world
  authorize-security-group-ingress "${MASTER_SG_ID}" "--protocol tcp --port 22 --cidr 0.0.0.0/0"
  authorize-security-group-ingress "${MINION_SG_ID}" "--protocol tcp --port 22 --cidr 0.0.0.0/0"

  # HTTPS to the master is allowed (for API access)
  authorize-security-group-ingress "${MASTER_SG_ID}" "--protocol tcp --port 443 --cidr 0.0.0.0/0"

  # Get or create master persistent volume
  ensure-master-pd

  # Determine extra certificate names for master
  octets=($(echo "$SERVICE_CLUSTER_IP_RANGE" | sed -e 's|/.*||' -e 's/\./ /g'))
  ((octets[3]+=1))
  service_ip=$(echo "${octets[*]}" | sed 's/ /./g')
  MASTER_EXTRA_SANS="IP:${service_ip},DNS:kubernetes,DNS:kubernetes.default,DNS:kubernetes.default.svc,DNS:kubernetes.default.svc.${DNS_DOMAIN},DNS:${MASTER_NAME}"


  (
    # We pipe this to the ami as a startup script in the user-data field.  Requires a compatible ami
    echo "#! /bin/bash"
    echo "mkdir -p /var/cache/kubernetes-install"
    echo "cd /var/cache/kubernetes-install"
    echo "readonly SALT_MASTER='${MASTER_INTERNAL_IP}'"
    echo "readonly INSTANCE_PREFIX='${INSTANCE_PREFIX}'"
    echo "readonly NODE_INSTANCE_PREFIX='${NODE_INSTANCE_PREFIX}'"
    echo "readonly CLUSTER_IP_RANGE='${CLUSTER_IP_RANGE}'"
    echo "readonly ALLOCATE_NODE_CIDRS='${ALLOCATE_NODE_CIDRS}'"
    echo "readonly SERVER_BINARY_TAR_URL='${SERVER_BINARY_TAR_URL}'"
    echo "readonly SALT_TAR_URL='${SALT_TAR_URL}'"
    echo "readonly ZONE='${ZONE}'"
    echo "readonly KUBE_USER='${KUBE_USER}'"
    echo "readonly KUBE_PASSWORD='${KUBE_PASSWORD}'"
    echo "readonly SERVICE_CLUSTER_IP_RANGE='${SERVICE_CLUSTER_IP_RANGE}'"
    echo "readonly ENABLE_CLUSTER_MONITORING='${ENABLE_CLUSTER_MONITORING:-none}'"
    echo "readonly ENABLE_CLUSTER_LOGGING='${ENABLE_CLUSTER_LOGGING:-false}'"
    echo "readonly ENABLE_NODE_LOGGING='${ENABLE_NODE_LOGGING:-false}'"
    echo "readonly LOGGING_DESTINATION='${LOGGING_DESTINATION:-}'"
    echo "readonly ELASTICSEARCH_LOGGING_REPLICAS='${ELASTICSEARCH_LOGGING_REPLICAS:-}'"
    echo "readonly ENABLE_CLUSTER_DNS='${ENABLE_CLUSTER_DNS:-false}'"
    echo "readonly ENABLE_CLUSTER_UI='${ENABLE_CLUSTER_UI:-false}'"
    echo "readonly DNS_REPLICAS='${DNS_REPLICAS:-}'"
    echo "readonly DNS_SERVER_IP='${DNS_SERVER_IP:-}'"
    echo "readonly DNS_DOMAIN='${DNS_DOMAIN:-}'"
    echo "readonly ADMISSION_CONTROL='${ADMISSION_CONTROL:-}'"
    echo "readonly MASTER_IP_RANGE='${MASTER_IP_RANGE:-}'"
    echo "readonly KUBELET_TOKEN='${KUBELET_TOKEN}'"
    echo "readonly KUBE_PROXY_TOKEN='${KUBE_PROXY_TOKEN}'"
    echo "readonly DOCKER_STORAGE='${DOCKER_STORAGE:-}'"
    echo "readonly MASTER_EXTRA_SANS='${MASTER_EXTRA_SANS:-}'"
    grep -v "^#" "${KUBE_ROOT}/cluster/aws/templates/common.sh"
    grep -v "^#" "${KUBE_ROOT}/cluster/aws/templates/format-disks.sh"
    grep -v "^#" "${KUBE_ROOT}/cluster/aws/templates/setup-master-pd.sh"
    grep -v "^#" "${KUBE_ROOT}/cluster/aws/templates/create-dynamic-salt-files.sh"
    grep -v "^#" "${KUBE_ROOT}/cluster/aws/templates/download-release.sh"
    grep -v "^#" "${KUBE_ROOT}/cluster/aws/templates/salt-master.sh"
  ) > "${KUBE_TEMP}/master-start.sh"

  echo "Starting Master"
  master_id=$($AWS_CMD run-instances \
    --image-id $AWS_IMAGE \
    --iam-instance-profile Name=$IAM_PROFILE_MASTER \
    --instance-type $MASTER_SIZE \
    --subnet-id $SUBNET_ID \
    --private-ip-address $MASTER_INTERNAL_IP \
    --key-name ${AWS_SSH_KEY_NAME} \
    --security-group-ids ${MASTER_SG_ID} \
    --associate-public-ip-address \
    --block-device-mappings "${MASTER_BLOCK_DEVICE_MAPPINGS}" \
    --user-data file://${KUBE_TEMP}/master-start.sh | json_val '["Instances"][0]["InstanceId"]')
  add-tag $master_id Name $MASTER_NAME
  add-tag $master_id Role $MASTER_TAG
  add-tag $master_id KubernetesCluster ${CLUSTER_ID}

  echo "Waiting for master to be ready"
  local attempt=0

  while true; do
    echo -n Attempt "$(($attempt+1))" to check for master node
    local ip=$(get_instance_public_ip ${master_id})
    if [[ -z "${ip}" ]]; then
      if (( attempt > 30 )); then
        echo
        echo -e "${color_red}master failed to start. Your cluster is unlikely" >&2
        echo "to work correctly. Please run ./cluster/kube-down.sh and re-create the" >&2
        echo -e "cluster. (sorry!)${color_norm}" >&2
        exit 1
      fi
    else
      # We are not able to add an elastic ip, a route or volume to the instance until that instance is in "running" state.
      wait-for-instance-running $master_id

      KUBE_MASTER=${MASTER_NAME}
      KUBE_MASTER_IP=$(assign-elastic-ip $ip $master_id)
      echo -e " ${color_green}[master running @${KUBE_MASTER_IP}]${color_norm}"

      # This is a race between instance start and volume attachment.  There appears to be no way to start an AWS instance with a volume attached.
      # To work around this, we wait for volume to be ready in setup-master-pd.sh
      echo "Attaching persistent data volume (${MASTER_DISK_ID}) to master"
      $AWS_CMD attach-volume --volume-id ${MASTER_DISK_ID} --device /dev/sdb --instance-id ${master_id}

      sleep 10
      $AWS_CMD create-route --route-table-id $ROUTE_TABLE_ID --destination-cidr-block ${MASTER_IP_RANGE} --instance-id $master_id > $LOG

      break
    fi
    echo -e " ${color_yellow}[master not working yet]${color_norm}"
    attempt=$(($attempt+1))
    sleep 10
  done

  # Check for SSH connectivity
  attempt=0
  while true; do
    echo -n Attempt "$(($attempt+1))" to check for SSH to master
    local output
    local ok=1
    output=$(ssh -oStrictHostKeyChecking=no -i "${AWS_SSH_KEY}" ${SSH_USER}@${KUBE_MASTER_IP} uptime 2> $LOG) || ok=0
    if [[ ${ok} == 0 ]]; then
      if (( attempt > 30 )); then
        echo
        echo "(Failed) output was: ${output}"
        echo
        echo -e "${color_red}Unable to ssh to master on ${KUBE_MASTER_IP}. Your cluster is unlikely" >&2
        echo "to work correctly. Please run ./cluster/kube-down.sh and re-create the" >&2
        echo -e "cluster. (sorry!)${color_norm}" >&2
        exit 1
      fi
    else
      echo -e " ${color_green}[ssh to master working]${color_norm}"
      break
    fi
    echo -e " ${color_yellow}[ssh to master not working yet]${color_norm}"
    attempt=$(($attempt+1))
    sleep 10
  done

  # We need the salt-master to be up for the minions to work
  attempt=0
  while true; do
    echo -n Attempt "$(($attempt+1))" to check for salt-master
    local output
    local ok=1
    output=$(ssh -oStrictHostKeyChecking=no -i "${AWS_SSH_KEY}" ${SSH_USER}@${KUBE_MASTER_IP} pgrep salt-master 2> $LOG) || ok=0
    if [[ ${ok} == 0 ]]; then
      if (( attempt > 30 )); then
        echo
        echo "(Failed) output was: ${output}"
        echo
        echo -e "${color_red}salt-master failed to start on ${KUBE_MASTER_IP}. Your cluster is unlikely" >&2
        echo "to work correctly. Please run ./cluster/kube-down.sh and re-create the" >&2
        echo -e "cluster. (sorry!)${color_norm}" >&2
        exit 1
      fi
    else
      echo -e " ${color_green}[salt-master running]${color_norm}"
      break
    fi
    echo -e " ${color_yellow}[salt-master not working yet]${color_norm}"
    attempt=$(($attempt+1))
    sleep 10
  done

  echo "Creating minion configuration"
  generate-minion-user-data > "${KUBE_TEMP}/minion-user-data"
  local public_ip_option
  if [[ "${ENABLE_MINION_PUBLIC_IP}" == "true" ]]; then
    public_ip_option="--associate-public-ip-address"
  else
    public_ip_option="--no-associate-public-ip-address"
  fi
  ${AWS_ASG_CMD} create-launch-configuration \
      --launch-configuration-name ${ASG_NAME} \
      --image-id $KUBE_MINION_IMAGE \
      --iam-instance-profile ${IAM_PROFILE_MINION} \
      --instance-type $MINION_SIZE \
      --key-name ${AWS_SSH_KEY_NAME} \
      --security-groups ${MINION_SG_ID} \
      ${public_ip_option} \
      --block-device-mappings "${MINION_BLOCK_DEVICE_MAPPINGS}" \
      --user-data "file://${KUBE_TEMP}/minion-user-data"

  echo "Creating autoscaling group"
  ${AWS_ASG_CMD} create-auto-scaling-group \
      --auto-scaling-group-name ${ASG_NAME} \
      --launch-configuration-name ${ASG_NAME} \
      --min-size ${NUM_MINIONS} \
      --max-size ${NUM_MINIONS} \
      --vpc-zone-identifier ${SUBNET_ID} \
      --tags ResourceId=${ASG_NAME},ResourceType=auto-scaling-group,Key=Name,Value=${NODE_INSTANCE_PREFIX} \
             ResourceId=${ASG_NAME},ResourceType=auto-scaling-group,Key=Role,Value=${MINION_TAG} \
             ResourceId=${ASG_NAME},ResourceType=auto-scaling-group,Key=KubernetesCluster,Value=${CLUSTER_ID}

  # Wait for the minions to be running
  # TODO(justinsb): This is really not needed any more
  attempt=0
  while true; do
    find-running-minions > $LOG
    if [[ ${#MINION_IDS[@]} == ${NUM_MINIONS} ]]; then
      echo -e " ${color_green}${#MINION_IDS[@]} minions started; ready${color_norm}"
      break
    fi

    if (( attempt > 30 )); then
      echo
      echo "Expected number of minions did not start in time"
      echo
      echo -e "${color_red}Expected number of minions failed to start.  Your cluster is unlikely" >&2
      echo "to work correctly. Please run ./cluster/kube-down.sh and re-create the" >&2
      echo -e "cluster. (sorry!)${color_norm}" >&2
      exit 1
    fi

    echo -e " ${color_yellow}${#MINION_IDS[@]} minions started; waiting${color_norm}"
    attempt=$(($attempt+1))
    sleep 10
  done

  detect-master > $LOG
  detect-minions > $LOG

  # TODO(justinsb): This is really not necessary any more
  # Wait 3 minutes for cluster to come up.  We hit it with a "highstate" after that to
  # make sure that everything is well configured.
  # TODO: Can we poll here?
  echo "Waiting 3 minutes for cluster to settle"
  local i
  for (( i=0; i < 6*3; i++)); do
    printf "."
    sleep 10
  done
  echo "Re-running salt highstate"
  ssh -oStrictHostKeyChecking=no -i "${AWS_SSH_KEY}" ${SSH_USER}@${KUBE_MASTER_IP} sudo salt '*' state.highstate > $LOG

  echo "Waiting for cluster initialization."
  echo
  echo "  This will continually check to see if the API for kubernetes is reachable."
  echo "  This might loop forever if there was some uncaught error during start"
  echo "  up."
  echo

  until $(curl --insecure --user ${KUBE_USER}:${KUBE_PASSWORD} --max-time 5 \
    --fail --output $LOG --silent https://${KUBE_MASTER_IP}/healthz); do
    printf "."
    sleep 2
  done

  echo "Kubernetes cluster created."

  # TODO use token instead of kube_auth
  export KUBE_CERT="/tmp/$RANDOM-kubecfg.crt"
  export KUBE_KEY="/tmp/$RANDOM-kubecfg.key"
  export CA_CERT="/tmp/$RANDOM-kubernetes.ca.crt"
  export CONTEXT="aws_${INSTANCE_PREFIX}"

  local kubectl="${KUBE_ROOT}/cluster/kubectl.sh"

  # TODO: generate ADMIN (and KUBELET) tokens and put those in the master's
  # config file.  Distribute the same way the htpasswd is done.
  (
    umask 077
    ssh -oStrictHostKeyChecking=no -i "${AWS_SSH_KEY}" "${SSH_USER}@${KUBE_MASTER_IP}" sudo cat /srv/kubernetes/kubecfg.crt >"${KUBE_CERT}" 2>"$LOG"
    ssh -oStrictHostKeyChecking=no -i "${AWS_SSH_KEY}" "${SSH_USER}@${KUBE_MASTER_IP}" sudo cat /srv/kubernetes/kubecfg.key >"${KUBE_KEY}" 2>"$LOG"
    ssh -oStrictHostKeyChecking=no -i "${AWS_SSH_KEY}" "${SSH_USER}@${KUBE_MASTER_IP}" sudo cat /srv/kubernetes/ca.crt >"${CA_CERT}" 2>"$LOG"

    create-kubeconfig
  )

  echo "Sanity checking cluster..."

  sleep 5

  # Don't bail on errors, we want to be able to print some info.
  set +e

  # Basic sanity checking
  # TODO(justinsb): This is really not needed any more
  local rc # Capture return code without exiting because of errexit bash option
  for (( i=0; i<${#KUBE_MINION_IP_ADDRESSES[@]}; i++)); do
      # Make sure docker is installed and working.
      local attempt=0
      while true; do
        local minion_ip=${KUBE_MINION_IP_ADDRESSES[$i]}
        echo -n "Attempt $(($attempt+1)) to check Docker on node @ ${minion_ip} ..."
        local output=`check-minion ${minion_ip}`
        echo $output
        if [[ "${output}" != "working" ]]; then
          if (( attempt > 9 )); then
            echo
            echo -e "${color_red}Your cluster is unlikely to work correctly." >&2
            echo "Please run ./cluster/kube-down.sh and re-create the" >&2
            echo -e "cluster. (sorry!)${color_norm}" >&2
            exit 1
          fi
        else
          break
        fi
        attempt=$(($attempt+1))
        sleep 30
      done
  done

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

function kube-down {
  local vpc_id=$(get_vpc_id)
  if [[ -n "${vpc_id}" ]]; then
    local elb_ids=$(get_elbs_in_vpc ${vpc_id})
    if [[ -n "${elb_ids}" ]]; then
      echo "Deleting ELBs in: ${vpc_id}"
      for elb_id in ${elb_ids}; do
        $AWS_ELB_CMD delete-load-balancer --load-balancer-name=${elb_id}
      done

      echo "Waiting for ELBs to be deleted"
      while true; do
        elb_ids=$(get_elbs_in_vpc ${vpc_id})
        if [[ -z "$elb_ids"  ]]; then
          echo "All ELBs deleted"
          break
        else
          echo "ELBs not yet deleted: $elb_ids"
          echo "Sleeping for 3 seconds..."
          sleep 3
        fi
      done
    fi

    if [[ -n $(${AWS_ASG_CMD} --output text describe-auto-scaling-groups --auto-scaling-group-names ${ASG_NAME} --query AutoScalingGroups[].AutoScalingGroupName) ]]; then
      echo "Deleting auto-scaling group: ${ASG_NAME}"
      ${AWS_ASG_CMD} delete-auto-scaling-group --force-delete --auto-scaling-group-name ${ASG_NAME}
    fi
    if [[ -n $(${AWS_ASG_CMD} --output text describe-launch-configurations --launch-configuration-names ${ASG_NAME} --query LaunchConfigurations[].LaunchConfigurationName) ]]; then
      echo "Deleting auto-scaling launch configuration: ${ASG_NAME}"
      ${AWS_ASG_CMD} delete-launch-configuration --launch-configuration-name ${ASG_NAME}
    fi

    echo "Deleting instances in VPC: ${vpc_id}"
    instance_ids=$($AWS_CMD --output text describe-instances \
                            --filters Name=vpc-id,Values=${vpc_id} \
                                      Name=tag:KubernetesCluster,Values=${CLUSTER_ID} \
                            --query Reservations[].Instances[].InstanceId)
    if [[ -n "${instance_ids}" ]]; then
      $AWS_CMD terminate-instances --instance-ids ${instance_ids} > $LOG
      echo "Waiting for instances to be deleted"
      while true; do
        local instance_states=$($AWS_CMD describe-instances --instance-ids ${instance_ids} | expect_instance_states terminated)
        if [[ -z "${instance_states}" ]]; then
          echo "All instances deleted"
          break
        else
          echo "Instances not yet deleted: ${instance_states}"
          echo "Sleeping for 3 seconds..."
          sleep 3
        fi
      done
    fi

    echo "Deleting VPC: ${vpc_id}"
    default_sg_id=$($AWS_CMD --output text describe-security-groups \
                             --filters Name=vpc-id,Values=${vpc_id} \
                                       Name=group-name,Values=default \
                             --query SecurityGroups[].GroupId \
                    | tr "\t" "\n")
    sg_ids=$($AWS_CMD --output text describe-security-groups \
                      --filters Name=vpc-id,Values=${vpc_id} \
                                Name=tag:KubernetesCluster,Values=${CLUSTER_ID} \
                      --query SecurityGroups[].GroupId \
             | tr "\t" "\n")
    # First delete any inter-security group ingress rules
    # (otherwise we get dependency violations)
    for sg_id in ${sg_ids}; do
      # EC2 doesn't let us delete the default security group
      if [[ "${sg_id}" == "${default_sg_id}" ]]; then
        continue
      fi

      echo "Cleaning up security group: ${sg_id}"
      other_sgids=$(aws ec2 describe-security-groups --group-id "${sg_id}" --query SecurityGroups[].IpPermissions[].UserIdGroupPairs[].GroupId --output text)
      for other_sgid in ${other_sgids}; do
        $AWS_CMD revoke-security-group-ingress --group-id "${sg_id}" --source-group "${other_sgid}" --protocol all > $LOG
      done
    done

    for sg_id in ${sg_ids}; do
      # EC2 doesn't let us delete the default security group
      if [[ "${sg_id}" == "${default_sg_id}" ]]; then
        continue
      fi

      echo "Deleting security group: ${sg_id}"
      $AWS_CMD delete-security-group --group-id ${sg_id} > $LOG
    done

    subnet_ids=$($AWS_CMD --output text describe-subnets \
                          --filters Name=vpc-id,Values=${vpc_id} \
                                    Name=tag:KubernetesCluster,Values=${CLUSTER_ID} \
                          --query Subnets[].SubnetId \
             | tr "\t" "\n")
    for subnet_id in ${subnet_ids}; do
      $AWS_CMD delete-subnet --subnet-id ${subnet_id} > $LOG
    done

    igw_ids=$($AWS_CMD --output text describe-internet-gateways \
                       --filters Name=attachment.vpc-id,Values=${vpc_id} \
                       --query InternetGateways[].InternetGatewayId \
             | tr "\t" "\n")
    for igw_id in ${igw_ids}; do
      $AWS_CMD detach-internet-gateway --internet-gateway-id $igw_id --vpc-id $vpc_id > $LOG
      $AWS_CMD delete-internet-gateway --internet-gateway-id $igw_id > $LOG
    done

    route_table_ids=$($AWS_CMD --output text describe-route-tables \
                               --filters Name=vpc-id,Values=$vpc_id \
                                         Name=route.destination-cidr-block,Values=0.0.0.0/0 \
                               --query RouteTables[].RouteTableId \
                      | tr "\t" "\n")
    for route_table_id in ${route_table_ids}; do
      $AWS_CMD delete-route --route-table-id $route_table_id --destination-cidr-block 0.0.0.0/0 > $LOG
    done
    route_table_ids=$($AWS_CMD --output text describe-route-tables \
                               --filters Name=vpc-id,Values=$vpc_id \
                                         Name=tag:KubernetesCluster,Values=${CLUSTER_ID} \
                               --query RouteTables[].RouteTableId \
                      | tr "\t" "\n")
    for route_table_id in ${route_table_ids}; do
      $AWS_CMD delete-route-table --route-table-id $route_table_id > $LOG
    done

    $AWS_CMD delete-vpc --vpc-id $vpc_id > $LOG
  fi
}

# Update a kubernetes cluster with latest source
function kube-push {
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
    grep -v "^#" "${KUBE_ROOT}/cluster/aws/templates/common.sh"
    grep -v "^#" "${KUBE_ROOT}/cluster/aws/templates/download-release.sh"
    echo "echo Executing configuration"
    echo "sudo salt '*' mine.update"
    echo "sudo salt --force-color '*' state.highstate"
  ) | ssh -oStrictHostKeyChecking=no -i "${AWS_SSH_KEY}" ${SSH_USER}@${KUBE_MASTER_IP} sudo bash

  get-kubeconfig-basicauth

  echo
  echo "Kubernetes cluster is running.  The master is running at:"
  echo
  echo "  https://${KUBE_MASTER_IP}"
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
# called from hack/e2e.go only when running -up (it is run after kube-up).
#
# Assumed vars:
#   Variables from config.sh
function test-setup {
  VPC_ID=$(get_vpc_id)
  detect-security-groups

  # Open up port 80 & 8080 so common containers on minions can be reached
  # TODO(roberthbailey): Remove this once we are no longer relying on hostPorts.
  authorize-security-group-ingress "${MINION_SG_ID}" "--protocol tcp --port 80 --cidr 0.0.0.0/0"
  authorize-security-group-ingress "${MINION_SG_ID}" "--protocol tcp --port 8080 --cidr 0.0.0.0/0"

  # Open up the NodePort range
  # TODO(justinsb): Move to main setup, if we decide whether we want to do this by default.
  authorize-security-group-ingress "${MINION_SG_ID}" "--protocol all --port 30000-32767 --cidr 0.0.0.0/0"

  echo "test-setup complete"
}

# Execute after running tests to perform any required clean-up. This is called
# from hack/e2e.go
function test-teardown {
  # (ingress rules will be deleted along with the security group)
  echo "Shutting down test cluster."
  "${KUBE_ROOT}/cluster/kube-down.sh"
}


# SSH to a node by name ($1) and run a command ($2).
function ssh-to-node {
  local node="$1"
  local cmd="$2"

  if [[ "${node}" == "${MASTER_NAME}" ]]; then
    node=$(get_instanceid_from_name ${MASTER_NAME})
    if [[ -z "${node-}" ]]; then
      echo "Could not detect Kubernetes master node.  Make sure you've launched a cluster with 'kube-up.sh'"
      exit 1
    fi
  fi

  local ip=$(get_instance_public_ip ${node})
  if [[ -z "$ip" ]]; then
    echo "Could not detect IP for ${node}."
    exit 1
  fi

  for try in $(seq 1 5); do
    if ssh -oLogLevel=quiet -oStrictHostKeyChecking=no -i "${AWS_SSH_KEY}" ${SSH_USER}@${ip} "${cmd}"; then
      break
    fi
  done
}

# Restart the kube-proxy on a node ($1)
function restart-kube-proxy {
  ssh-to-node "$1" "sudo /etc/init.d/kube-proxy restart"
}

# Restart the kube-apiserver on a node ($1)
function restart-apiserver {
  ssh-to-node "$1" "sudo /etc/init.d/kube-apiserver restart"
}

# Perform preparations required to run e2e tests
function prepare-e2e() {
  # (AWS runs detect-project, I don't think we need to anything)
  # Note: we can't print anything here, or else the test tools will break with the extra output
  return
}

function get-tokens() {
  KUBELET_TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
  KUBE_PROXY_TOKEN=$(dd if=/dev/urandom bs=128 count=1 2>/dev/null | base64 | tr -d "=+/" | dd bs=32 count=1 2>/dev/null)
}
