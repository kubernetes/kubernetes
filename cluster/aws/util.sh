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

# This removes the final character in bash (somehow)
AWS_REGION=${ZONE%?}

export AWS_DEFAULT_REGION=${AWS_REGION}
AWS_CMD="aws --output json ec2"
AWS_ELB_CMD="aws --output json elb"

INTERNAL_IP_BASE=172.20.0
MASTER_IP_SUFFIX=.9
MASTER_INTERNAL_IP=${INTERNAL_IP_BASE}${MASTER_IP_SUFFIX}

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

function get_cidr {
  python -c "import json,sys; lst = [str(subnet['CidrBlock']) for subnet in json.load(sys.stdin)['Subnets'] if subnet['VpcId'] == '$1' and subnet['AvailabilityZone'] == '$2']; print ''.join(lst)"
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

function get_instance_public_ip {
  local tagName=$1
  $AWS_CMD --output text describe-instances \
    --filters Name=tag:Name,Values=${tagName} \
              Name=instance-state-name,Values=running \
              Name=tag:KubernetesCluster,Values=${CLUSTER_ID} \
    --query Reservations[].Instances[].NetworkInterfaces[0].Association.PublicIp
}


function detect-master () {
  KUBE_MASTER=${MASTER_NAME}
  if [[ -z "${KUBE_MASTER_IP-}" ]]; then
    KUBE_MASTER_IP=$(get_instance_public_ip $MASTER_NAME)
  fi
  if [[ -z "${KUBE_MASTER_IP-}" ]]; then
    echo "Could not detect Kubernetes master node.  Make sure you've launched a cluster with 'kube-up.sh'"
    exit 1
  fi
  echo "Using master: $KUBE_MASTER (external IP: $KUBE_MASTER_IP)"
}

function detect-minions () {
  KUBE_MINION_IP_ADDRESSES=()
  for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
    local minion_ip=$(get_instance_public_ip ${MINION_NAMES[$i]})
    echo "Found ${MINION_NAMES[$i]} at ${minion_ip}"
    KUBE_MINION_IP_ADDRESSES+=("${minion_ip}")
  done
  if [[ -z "$KUBE_MINION_IP_ADDRESSES" ]]; then
    echo "Could not detect Kubernetes minion nodes.  Make sure you've launched a cluster with 'kube-up.sh'"
    exit 1
  fi
}

# Detects the AMI to use (considering the region)
#
# Vars set:
#   AWS_IMAGE
function detect-image () {
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
  if ! aws s3 ls "s3://${AWS_S3_BUCKET}" > /dev/null 2>&1 ; then
    echo "Creating ${AWS_S3_BUCKET}"

    # Buckets must be globally uniquely named, so always create in a known region
    # We default to us-east-1 because that's the canonical region for S3,
    # and then the bucket is most-simply named (s3.amazonaws.com)
    aws s3 mb "s3://${AWS_S3_BUCKET}" --region ${AWS_S3_REGION}
  fi

  local s3_bucket_location=$(aws --output text s3api get-bucket-location --bucket ${AWS_S3_BUCKET})
  local s3_url_base=https://s3-${s3_bucket_location}.amazonaws.com
  if [[ "${s3_bucket_location}" == "None" ]]; then
    # "US Classic" does not follow the pattern
    s3_url_base=https://s3.amazonaws.com
  fi

  local -r staging_path="devel"

  echo "+++ Staging server tars to S3 Storage: ${AWS_S3_BUCKET}/${staging_path}"

  local server_binary_path="${staging_path}/${SERVER_BINARY_TAR##*/}"
  aws s3 cp "${SERVER_BINARY_TAR}" "s3://${AWS_S3_BUCKET}/${server_binary_path}"
  aws s3api put-object-acl --bucket ${AWS_S3_BUCKET} --key "${server_binary_path}" --grant-read 'uri="http://acs.amazonaws.com/groups/global/AllUsers"'
  SERVER_BINARY_TAR_URL="${s3_url_base}/${AWS_S3_BUCKET}/${server_binary_path}"

  local salt_tar_path="${staging_path}/${SALT_TAR##*/}"
  aws s3 cp "${SALT_TAR}" "s3://${AWS_S3_BUCKET}/${salt_tar_path}"
  aws s3api put-object-acl --bucket ${AWS_S3_BUCKET} --key "${salt_tar_path}" --grant-read 'uri="http://acs.amazonaws.com/groups/global/AllUsers"'
  SALT_TAR_URL="${s3_url_base}/${AWS_S3_BUCKET}/${salt_tar_path}"
}


# Ensure that we have a password created for validating to the master.  Will
# read from kubeconfig for the current context if available.
#
# Assumed vars
#   KUBE_ROOT
#
# Vars set:
#   KUBE_USER
#   KUBE_PASSWORD
function get-password {
  get-kubeconfig-basicauth
  if [[ -z "${KUBE_USER}" || -z "${KUBE_PASSWORD}" ]]; then
    KUBE_USER=admin
    KUBE_PASSWORD=$(python -c 'import string,random; print "".join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(16))')
  fi
}

# Adds a tag to an AWS resource
# usage: add-tag <resource-id> <tag-name> <tag-value>
function add-tag {
  echo "Adding tag to ${1}: ${2}=${3}"

  # We need to retry in case the resource isn't yet fully created
  sleep 3
  n=0
  until [ $n -ge 5 ]; do
    $AWS_CMD create-tags --resources ${1} --tags Key=${2},Value=${3} > $LOG && return
    n=$[$n+1]
    sleep 15
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
    instance_state=$($AWS_CMD describe-instances --instance-ids $instance_id | expect_instance_states running)
    if [[ "$instance_state" == "" ]]; then
      break
    else
      echo "Waiting for instance ${instance_id} to spawn"
      echo "Sleeping for 3 seconds..."
      sleep 3
    fi
  done
}

function kube-up {
  find-release-tars
  upload-server-tars

  ensure-temp-dir

  ensure-iam-profiles

  get-password

  if [[ ! -f "$AWS_SSH_KEY" ]]; then
    ssh-keygen -f "$AWS_SSH_KEY" -N ''
  fi

  detect-image

  $AWS_CMD import-key-pair --key-name kubernetes --public-key-material "file://$AWS_SSH_KEY.pub" > $LOG 2>&1 || true

  VPC_ID=$(get_vpc_id)

  if [[ -z "$VPC_ID" ]]; then
	  echo "Creating vpc."
	  VPC_ID=$($AWS_CMD create-vpc --cidr-block $INTERNAL_IP_BASE.0/16 | json_val '["Vpc"]["VpcId"]')
	  $AWS_CMD modify-vpc-attribute --vpc-id $VPC_ID --enable-dns-support '{"Value": true}' > $LOG
	  $AWS_CMD modify-vpc-attribute --vpc-id $VPC_ID --enable-dns-hostnames '{"Value": true}' > $LOG
	  add-tag $VPC_ID Name kubernetes-vpc
	  add-tag $VPC_ID KubernetesCluster ${CLUSTER_ID}
  fi

  echo "Using VPC $VPC_ID"

  SUBNET_ID=$($AWS_CMD describe-subnets | get_subnet_id $VPC_ID $ZONE)
  if [[ -z "$SUBNET_ID" ]]; then
    echo "Creating subnet."
    SUBNET_ID=$($AWS_CMD create-subnet --cidr-block $INTERNAL_IP_BASE.0/24 --vpc-id $VPC_ID --availability-zone ${ZONE} | json_val '["Subnet"]["SubnetId"]')
  else
    EXISTING_CIDR=$($AWS_CMD describe-subnets | get_cidr $VPC_ID $ZONE)
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
  ROUTE_TABLE_ID=$($AWS_CMD describe-route-tables --filters Name=vpc-id,Values=$VPC_ID | json_val '["RouteTables"][0]["RouteTableId"]')
  $AWS_CMD associate-route-table --route-table-id $ROUTE_TABLE_ID --subnet-id $SUBNET_ID > $LOG || true
  echo "Configuring route table."
  $AWS_CMD describe-route-tables --filters Name=vpc-id,Values=$VPC_ID > $LOG || true
  echo "Adding route to route table."
  $AWS_CMD create-route --route-table-id $ROUTE_TABLE_ID --destination-cidr-block 0.0.0.0/0 --gateway-id $IGW_ID > $LOG || true

  echo "Using Route Table $ROUTE_TABLE_ID"

  SEC_GROUP_ID=$($AWS_CMD --output text describe-security-groups \
                          --filters Name=vpc-id,Values=$VPC_ID \
                                    Name=group-name,Values=kubernetes-sec-group \
                          --query SecurityGroups[].GroupId \
                    | tr "\t" "\n")

  if [[ -z "$SEC_GROUP_ID" ]]; then
	  echo "Creating security group."
	  SEC_GROUP_ID=$($AWS_CMD create-security-group --group-name kubernetes-sec-group --description kubernetes-sec-group --vpc-id $VPC_ID | json_val '["GroupId"]')
	  $AWS_CMD authorize-security-group-ingress --group-id $SEC_GROUP_ID --protocol -1 --port all --cidr 0.0.0.0/0 > $LOG
  fi

  (
    # We pipe this to the ami as a startup script in the user-data field.  Requires a compatible ami
    echo "#! /bin/bash"
    echo "mkdir -p /var/cache/kubernetes-install"
    echo "cd /var/cache/kubernetes-install"
    echo "readonly SALT_MASTER='${MASTER_INTERNAL_IP}'"
    echo "readonly INSTANCE_PREFIX='${INSTANCE_PREFIX}'"
    echo "readonly NODE_INSTANCE_PREFIX='${INSTANCE_PREFIX}-minion'"
    echo "readonly SERVER_BINARY_TAR_URL='${SERVER_BINARY_TAR_URL}'"
    echo "readonly SALT_TAR_URL='${SALT_TAR_URL}'"
    echo "readonly ZONE='${ZONE}'"
    echo "readonly KUBE_USER='${KUBE_USER}'"
    echo "readonly KUBE_PASSWORD='${KUBE_PASSWORD}'"
    echo "readonly PORTAL_NET='${PORTAL_NET}'"
    echo "readonly ENABLE_CLUSTER_MONITORING='${ENABLE_CLUSTER_MONITORING:-false}'"
    echo "readonly ENABLE_NODE_MONITORING='${ENABLE_NODE_MONITORING:-false}'"
    echo "readonly ENABLE_CLUSTER_LOGGING='${ENABLE_CLUSTER_LOGGING:-false}'"
    echo "readonly ENABLE_NODE_LOGGING='${ENABLE_NODE_LOGGING:-false}'"
    echo "readonly LOGGING_DESTINATION='${LOGGING_DESTINATION:-}'"
    echo "readonly ELASTICSEARCH_LOGGING_REPLICAS='${ELASTICSEARCH_LOGGING_REPLICAS:-}'"
    echo "readonly ENABLE_CLUSTER_DNS='${ENABLE_CLUSTER_DNS:-false}'"
    echo "readonly DNS_REPLICAS='${DNS_REPLICAS:-}'"
    echo "readonly DNS_SERVER_IP='${DNS_SERVER_IP:-}'"
    echo "readonly DNS_DOMAIN='${DNS_DOMAIN:-}'"
    echo "readonly ADMISSION_CONTROL='${ADMISSION_CONTROL:-}'"
    echo "readonly MASTER_IP_RANGE='${MASTER_IP_RANGE:-}'"
    grep -v "^#" "${KUBE_ROOT}/cluster/aws/templates/common.sh"
    grep -v "^#" "${KUBE_ROOT}/cluster/aws/templates/format-disks.sh"
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
    --key-name kubernetes \
    --security-group-ids $SEC_GROUP_ID \
    --associate-public-ip-address \
    --user-data file://${KUBE_TEMP}/master-start.sh | json_val '["Instances"][0]["InstanceId"]')
  add-tag $master_id Name $MASTER_NAME
  add-tag $master_id Role $MASTER_TAG
  add-tag $master_id KubernetesCluster ${CLUSTER_ID}

  echo "Waiting for master to be ready"

  local attempt=0

   while true; do
    echo -n Attempt "$(($attempt+1))" to check for master node
    local ip=$(get_instance_public_ip $MASTER_NAME)
    if [[ -z "${ip}" ]]; then
      if (( attempt > 30 )); then
        echo
        echo -e "${color_red}master failed to start. Your cluster is unlikely" >&2
        echo "to work correctly. Please run ./cluster/kube-down.sh and re-create the" >&2
        echo -e "cluster. (sorry!)${color_norm}" >&2
        exit 1
      fi
    else
      KUBE_MASTER=${MASTER_NAME}
      KUBE_MASTER_IP=${ip}
      echo -e " ${color_green}[master running @${KUBE_MASTER_IP}]${color_norm}"

      # We are not able to add a route to the instance until that instance is in "running" state.
      wait-for-instance-running $master_id
      sleep 10
      $AWS_CMD create-route --route-table-id $ROUTE_TABLE_ID --destination-cidr-block ${MASTER_IP_RANGE} --instance-id $master_id > $LOG

      break
    fi
    echo -e " ${color_yellow}[master not working yet]${color_norm}"
    attempt=$(($attempt+1))
    sleep 10
  done

  # We need the salt-master to be up for the minions to work
  attempt=0
  while true; do
    echo -n Attempt "$(($attempt+1))" to check for salt-master
    local output
    output=$(ssh -oStrictHostKeyChecking=no -i "${AWS_SSH_KEY}" ubuntu@${KUBE_MASTER_IP} pgrep salt-master 2> $LOG) || output=""
    if [[ -z "${output}" ]]; then
      if (( attempt > 30 )); then
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

  MINION_IDS=()
  for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
    echo "Starting Minion (${MINION_NAMES[$i]})"
    (
      # We pipe this to the ami as a startup script in the user-data field.  Requires a compatible ami
      echo "#! /bin/bash"
      echo "SALT_MASTER='${MASTER_INTERNAL_IP}'"
      echo "MINION_IP_RANGE='${MINION_IP_RANGES[$i]}'"
      echo "DOCKER_OPTS='${EXTRA_DOCKER_OPTS:-}'"
      grep -v "^#" "${KUBE_ROOT}/cluster/aws/templates/format-disks.sh"
      grep -v "^#" "${KUBE_ROOT}/cluster/aws/templates/salt-minion.sh"
    ) > "${KUBE_TEMP}/minion-start-${i}.sh"
    minion_id=$($AWS_CMD run-instances \
      --image-id $AWS_IMAGE \
      --iam-instance-profile Name=$IAM_PROFILE_MINION \
      --instance-type $MINION_SIZE \
      --subnet-id $SUBNET_ID \
      --private-ip-address $INTERNAL_IP_BASE.1${i} \
      --key-name kubernetes \
      --security-group-ids $SEC_GROUP_ID \
      --associate-public-ip-address \
      --user-data file://${KUBE_TEMP}/minion-start-${i}.sh | json_val '["Instances"][0]["InstanceId"]')

    add-tag $minion_id Name ${MINION_NAMES[$i]}
    add-tag $minion_id Role $MINION_TAG
    add-tag $minion_id KubernetesCluster ${CLUSTER_ID}

    MINION_IDS[$i]=$minion_id
  done

  # Add routes to minions
  for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
    # We are not able to add a route to the instance until that instance is in "running" state.
    # This is quite an ugly solution to this problem. In Bash 4 we could use assoc. arrays to do this for
    # all instances at once but we can't be sure we are running Bash 4.
    minion_id=${MINION_IDS[$i]}
    wait-for-instance-running $minion_id
    echo "Minion ${MINION_NAMES[$i]} running"
    sleep 10
    $AWS_CMD modify-instance-attribute --instance-id $minion_id --source-dest-check '{"Value": false}' > $LOG
    $AWS_CMD create-route --route-table-id $ROUTE_TABLE_ID --destination-cidr-block ${MINION_IP_RANGES[$i]} --instance-id $minion_id > $LOG
  done

  FAIL=0
  for job in `jobs -p`; do
    wait $job || let "FAIL+=1"
  done
  if (( $FAIL != 0 )); then
    echo "${FAIL} commands failed.  Exiting."
    exit 2
  fi

  detect-master > $LOG
  detect-minions > $LOG

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
  ssh -oStrictHostKeyChecking=no -i "${AWS_SSH_KEY}" ubuntu@${KUBE_MASTER_IP} sudo salt '*' state.highstate > $LOG

  echo "Waiting for cluster initialization."
  echo
  echo "  This will continually check to see if the API for kubernetes is reachable."
  echo "  This might loop forever if there was some uncaught error during start"
  echo "  up."
  echo

  until $(curl --insecure --user ${KUBE_USER}:${KUBE_PASSWORD} --max-time 5 \
    --fail --output $LOG --silent https://${KUBE_MASTER_IP}/api/v1beta1/pods); do
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
    ssh -oStrictHostKeyChecking=no -i "${AWS_SSH_KEY}" "ubuntu@${KUBE_MASTER_IP}" sudo cat /srv/kubernetes/kubecfg.crt >"${KUBE_CERT}" 2>"$LOG"
    ssh -oStrictHostKeyChecking=no -i "${AWS_SSH_KEY}" "ubuntu@${KUBE_MASTER_IP}" sudo cat /srv/kubernetes/kubecfg.key >"${KUBE_KEY}" 2>"$LOG"
    ssh -oStrictHostKeyChecking=no -i "${AWS_SSH_KEY}" "ubuntu@${KUBE_MASTER_IP}" sudo cat /srv/kubernetes/ca.crt >"${CA_CERT}" 2>"$LOG"

    create-kubeconfig
  )

  echo "Sanity checking cluster..."

  sleep 5

  # Don't bail on errors, we want to be able to print some info.
  set +e

  # Basic sanity checking
  local rc # Capture return code without exiting because of errexit bash option
  for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
      # Make sure docker is installed and working.
      local attempt=0
      while true; do
        local minion_name=${MINION_NAMES[$i]}
        local minion_ip=${KUBE_MINION_IP_ADDRESSES[$i]}
        echo -n Attempt "$(($attempt+1))" to check Docker on node "${minion_name} @ ${minion_ip}" ...
        local output=$(ssh -oStrictHostKeyChecking=no -i "${AWS_SSH_KEY}" ubuntu@$minion_ip sudo docker ps -a 2>/dev/null)
        if [[ -z "${output}" ]]; then
          if (( attempt > 9 )); then
            echo
            echo -e "${color_red}Docker failed to install on node ${minion_name}. Your cluster is unlikely" >&2
            echo "to work correctly. Please run ./cluster/kube-down.sh and re-create the" >&2
            echo -e "cluster. (sorry!)${color_norm}" >&2
            exit 1
          fi
        # TODO: Reintroduce this (where does this container come from?)
#        elif [[ "${output}" != *"kubernetes/pause"* ]]; then
#          if (( attempt > 9 )); then
#            echo
#            echo -e "${color_red}Failed to observe kubernetes/pause on node ${minion_name}. Your cluster is unlikely" >&2
#            echo "to work correctly. Please run ./cluster/kube-down.sh and re-create the" >&2
#            echo -e "cluster. (sorry!)${color_norm}" >&2
#            exit 1
#          fi
        else
          echo -e " ${color_green}[working]${color_norm}"
          break
        fi
        echo -e " ${color_yellow}[not working yet]${color_norm}"
        # Start Docker, in case it failed to start.
        ssh -oStrictHostKeyChecking=no -i "${AWS_SSH_KEY}" ubuntu@$minion_ip sudo service docker start > $LOG 2>&1
        attempt=$(($attempt+1))
        sleep 30
      done
  done

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
                             --filters Name=vpc-id,Values=${vpc_id} Name=group-name,Values=default \
                             --query SecurityGroups[].GroupId \
                    | tr "\t" "\n")
    sg_ids=$($AWS_CMD --output text describe-security-groups \
                      --filters Name=vpc-id,Values=${vpc_id} \
                      --query SecurityGroups[].GroupId \
             | tr "\t" "\n")
    for sg_id in ${sg_ids}; do
      # EC2 doesn't let us delete the default security group
      if [[ "${sg_id}" != "${default_sg_id}" ]]; then
        $AWS_CMD delete-security-group --group-id ${sg_id} > $LOG
      fi
    done

    subnet_ids=$($AWS_CMD --output text describe-subnets \
                          --filters Name=vpc-id,Values=${vpc_id} \
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
  ) | ssh -oStrictHostKeyChecking=no -i "${AWS_SSH_KEY}" ubuntu@${KUBE_MASTER_IP} sudo bash

  get-password

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
  echo "test-setup complete"
}

# Execute after running tests to perform any required clean-up. This is called
# from hack/e2e.go
function test-teardown {
  echo "Shutting down test cluster."
  "${KUBE_ROOT}/cluster/kube-down.sh"
}


# SSH to a node by name ($1) and run a command ($2).
function ssh-to-node {
  local node="$1"
  local cmd="$2"

  local ip=$(get_instance_public_ip ${node})
  if [[ -z "$ip" ]]; then
    echo "Could not detect IP for ${node}."
    exit 1
  fi

  for try in $(seq 1 5); do
    if ssh -oLogLevel=quiet -oStrictHostKeyChecking=no -i "${AWS_SSH_KEY}" ubuntu@${ip} "${cmd}"; then
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
