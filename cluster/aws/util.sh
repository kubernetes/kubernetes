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
source "${KUBE_ROOT}/cluster/aws/${KUBE_CONFIG_FILE-"config-default.sh"}"

export AWS_DEFAULT_REGION=${ZONE}
AWS_CMD="aws --output json ec2"

MASTER_INTERNAL_IP=172.20.0.9

function json_val {
    python -c 'import json,sys;obj=json.load(sys.stdin);print obj'$1''
}

# TODO (ayurchuk) Refactor the get_* functions to use filters
# TODO (bburns) Parameterize this for multiple cluster per project
function get_instance_ids {
  python -c "import json,sys; lst = [str(instance['InstanceId']) for reservation in json.load(sys.stdin)['Reservations'] for instance in reservation['Instances'] for tag in instance.get('Tags', []) if tag['Value'].startswith('${MASTER_TAG}') or tag['Value'].startswith('${MINION_TAG}')]; print ' '.join(lst)"
}

function get_vpc_id {
  python -c 'import json,sys; lst = [str(vpc["VpcId"]) for vpc in json.load(sys.stdin)["Vpcs"] for tag in vpc.get("Tags", []) if tag["Value"] == "kubernetes-vpc"]; print "".join(lst)'
}

function get_subnet_id {
  python -c "import json,sys; lst = [str(subnet['SubnetId']) for subnet in json.load(sys.stdin)['Subnets'] if subnet['VpcId'] == '$1']; print ''.join(lst)"
}

function get_igw_id {
  python -c "import json,sys; lst = [str(igw['InternetGatewayId']) for igw in json.load(sys.stdin)['InternetGateways'] for attachment in igw['Attachments'] if attachment['VpcId'] == '$1']; print ''.join(lst)"
}

function get_route_table_id {
  python -c "import json,sys; lst = [str(route_table['RouteTableId']) for route_table in json.load(sys.stdin)['RouteTables'] if route_table['VpcId'] == '$1']; print ''.join(lst)"
}

function get_sec_group_id {
  python -c 'import json,sys; lst = [str(group["GroupId"]) for group in json.load(sys.stdin)["SecurityGroups"] if group["GroupName"] == "kubernetes-sec-group"]; print "".join(lst)'
}

function expect_instance_states {
  python -c "import json,sys; lst = [str(instance['InstanceId']) for reservation in json.load(sys.stdin)['Reservations'] for instance in reservation['Instances'] if instance['State']['Name'] != '$1']; print ' '.join(lst)"
}

function get_instance_public_ip {
  local tagName=$1
  $AWS_CMD --output text describe-instances \
    --filters Name=tag:Name,Values=${tagName} Name=instance-state-name,Values=running \
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
  if [ -z "$KUBE_MINION_IP_ADDRESSES" ]; then
    echo "Could not detect Kubernetes minion nodes.  Make sure you've launched a cluster with 'kube-up.sh'"
    exit 1
  fi
}

# Verify prereqs
function verify-prereqs {
  if [ "$(which aws)" == "" ]; then
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

  local project_hash=
  local key=$(aws configure get aws_access_key_id)
  if which md5 > /dev/null 2>&1; then
    project_hash=$(md5 -q -s "${USER} ${key}")
  else
    project_hash=$(echo -n "${USER} ${key}" | md5sum | awk '{ print $1 }')
  fi
  local -r staging_bucket="kubernetes-staging-${project_hash}"

  echo "Uploading to Amazon S3"
  if ! aws s3 ls "s3://${staging_bucket}" > /dev/null 2>&1 ; then
    echo "Creating ${staging_bucket}"
    aws s3 mb "s3://${staging_bucket}"
  fi

  aws s3api put-bucket-acl --bucket $staging_bucket --acl public-read

  local -r staging_path="${staging_bucket}/devel"

  echo "+++ Staging server tars to S3 Storage: ${staging_path}"
  SERVER_BINARY_TAR_URL="${staging_path}/${SERVER_BINARY_TAR##*/}"
  aws s3 cp "${SERVER_BINARY_TAR}" "s3://${SERVER_BINARY_TAR_URL}"
  aws s3api put-object-acl --bucket ${staging_bucket} --key "devel/${SERVER_BINARY_TAR##*/}" --grant-read 'uri="http://acs.amazonaws.com/groups/global/AllUsers"'
  SALT_TAR_URL="${staging_path}/${SALT_TAR##*/}"
  aws s3 cp "${SALT_TAR}" "s3://${SALT_TAR_URL}"
  aws s3api put-object-acl --bucket ${staging_bucket} --key "devel/${SALT_TAR##*/}" --grant-read 'uri="http://acs.amazonaws.com/groups/global/AllUsers"'
}


# Ensure that we have a password created for validating to the master.  Will
# read from the kubernetes auth-file for the current context if available.
#
# Assumed vars
#   KUBE_ROOT
#
# Vars set:
#   KUBE_USER
#   KUBE_PASSWORD
function get-password {
  # go template to extract the auth-path of the current-context user
  # Note: we save dot ('.') to $dot because the 'with' action overrides dot
  local template='{{$dot := .}}{{with $ctx := index $dot "current-context"}}{{$user := index $dot "contexts" $ctx "user"}}{{index $dot "users" $user "auth-path"}}{{end}}'
  local file=$("${KUBE_ROOT}/cluster/kubectl.sh" config view -o template --template="${template}")
  if [[ ! -z "$file" && -r "$file" ]]; then
    KUBE_USER=$(cat "$file" | python -c 'import json,sys;print json.load(sys.stdin)["User"]')
    KUBE_PASSWORD=$(cat "$file" | python -c 'import json,sys;print json.load(sys.stdin)["Password"]')
    return
  fi
  KUBE_USER=admin
  KUBE_PASSWORD=$(python -c 'import string,random; print "".join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(16))')
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

function kube-up {
  find-release-tars
  upload-server-tars

  ensure-temp-dir

  get-password
  python "${KUBE_ROOT}/third_party/htpasswd/htpasswd.py" \
    -b -c "${KUBE_TEMP}/htpasswd" "$KUBE_USER" "$KUBE_PASSWORD"
  local htpasswd
  htpasswd=$(cat "${KUBE_TEMP}/htpasswd")

  if [ ! -f $AWS_SSH_KEY ]; then
    ssh-keygen -f $AWS_SSH_KEY -N ''
  fi

  aws iam get-instance-profile --instance-profile-name ${IAM_PROFILE} || {
        echo "You need to set up an IAM profile and role for kubernetes"
        exit 1
  }

  $AWS_CMD import-key-pair --key-name kubernetes --public-key-material file://$AWS_SSH_KEY.pub > $LOG 2>&1 || true

  VPC_ID=$($AWS_CMD describe-vpcs | get_vpc_id)

  if [ -z "$VPC_ID" ]; then
	  echo "Creating vpc."
	  VPC_ID=$($AWS_CMD create-vpc --cidr-block 172.20.0.0/16 | json_val '["Vpc"]["VpcId"]')
	  $AWS_CMD modify-vpc-attribute --vpc-id $VPC_ID --enable-dns-support '{"Value": true}' > $LOG
	  $AWS_CMD modify-vpc-attribute --vpc-id $VPC_ID --enable-dns-hostnames '{"Value": true}' > $LOG
	  add-tag $VPC_ID Name kubernetes-vpc
  fi

  echo "Using VPC $VPC_ID"

  SUBNET_ID=$($AWS_CMD describe-subnets | get_subnet_id $VPC_ID)
  if [ -z "$SUBNET_ID" ]; then
	  echo "Creating subnet."
	  SUBNET_ID=$($AWS_CMD create-subnet --cidr-block 172.20.0.0/24 --vpc-id $VPC_ID | json_val '["Subnet"]["SubnetId"]')
  fi

  echo "Using subnet $SUBNET_ID"

  IGW_ID=$($AWS_CMD describe-internet-gateways | get_igw_id $VPC_ID)
  if [ -z "$IGW_ID" ]; then
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

  SEC_GROUP_ID=$($AWS_CMD describe-security-groups | get_sec_group_id)

  if [ -z "$SEC_GROUP_ID" ]; then
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
    echo "readonly SERVER_BINARY_TAR_URL='https://s3-${ZONE}.amazonaws.com/${SERVER_BINARY_TAR_URL}'"
    echo "readonly SALT_TAR_URL='https://s3-${ZONE}.amazonaws.com/${SALT_TAR_URL}'"
    echo "readonly AWS_ZONE='${ZONE}'"
    echo "readonly MASTER_HTPASSWD='${htpasswd}'"
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
    grep -v "^#" "${KUBE_ROOT}/cluster/aws/templates/common.sh"
    grep -v "^#" "${KUBE_ROOT}/cluster/aws/templates/create-dynamic-salt-files.sh"
    grep -v "^#" "${KUBE_ROOT}/cluster/aws/templates/download-release.sh"
    grep -v "^#" "${KUBE_ROOT}/cluster/aws/templates/salt-master.sh"
  ) > "${KUBE_TEMP}/master-start.sh"

  echo "Starting Master"
  master_id=$($AWS_CMD run-instances \
    --image-id $IMAGE \
    --iam-instance-profile Name=$IAM_PROFILE \
    --instance-type $MASTER_SIZE \
    --subnet-id $SUBNET_ID \
    --private-ip-address 172.20.0.9 \
    --key-name kubernetes \
    --security-group-ids $SEC_GROUP_ID \
    --associate-public-ip-address \
    --user-data file://${KUBE_TEMP}/master-start.sh | json_val '["Instances"][0]["InstanceId"]')
  add-tag $master_id Name $MASTER_NAME
  add-tag $master_id Role $MASTER_TAG

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
    output=$(ssh -oStrictHostKeyChecking=no -i ${AWS_SSH_KEY} ubuntu@${KUBE_MASTER_IP} pgrep salt-master 2> $LOG) || output=""
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


  for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
    echo "Starting Minion (${MINION_NAMES[$i]})"
    (
      # We pipe this to the ami as a startup script in the user-data field.  Requires a compatible ami
      echo "#! /bin/bash"
      echo "SALT_MASTER='${MASTER_INTERNAL_IP}'"
      echo "MINION_IP_RANGE='${MINION_IP_RANGES[$i]}'"
      grep -v "^#" "${KUBE_ROOT}/cluster/aws/templates/salt-minion.sh"
    ) > "${KUBE_TEMP}/minion-start-${i}.sh"
    minion_id=$($AWS_CMD run-instances \
      --image-id $IMAGE \
      --iam-instance-profile Name=$IAM_PROFILE \
      --instance-type $MINION_SIZE \
      --subnet-id $SUBNET_ID \
      --private-ip-address 172.20.0.1${i} \
      --key-name kubernetes \
      --security-group-ids $SEC_GROUP_ID \
      --associate-public-ip-address \
      --user-data file://${KUBE_TEMP}/minion-start-${i}.sh | json_val '["Instances"][0]["InstanceId"]')

    add-tag $minion_id Name ${MINION_NAMES[$i]}
    add-tag $minion_id Role $MINION_TAG

    sleep 3
    $AWS_CMD modify-instance-attribute --instance-id $minion_id --source-dest-check '{"Value": false}' > $LOG

    # We are not able to add a route to the instance until that instance is in "running" state.
    # This is quite an ugly solution to this problem. In Bash 4 we could use assoc. arrays to do this for
    # all instances at once but we can't be sure we are running Bash 4.
    while true; do
      instance_state=$($AWS_CMD describe-instances --instance-ids $minion_id | expect_instance_states running)
      if [[ "$instance_state" == "" ]]; then
        echo "Minion ${MINION_NAMES[$i]} running"
        sleep 10
        $AWS_CMD create-route --route-table-id $ROUTE_TABLE_ID --destination-cidr-block ${MINION_IP_RANGES[$i]} --instance-id $minion_id > $LOG
        break
      else
        echo "Waiting for minion ${MINION_NAMES[$i]} to spawn"
        echo "Sleeping for 3 seconds..."
        sleep 3
      fi
    done
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
  ssh -oStrictHostKeyChecking=no -i ${AWS_SSH_KEY} ubuntu@${KUBE_MASTER_IP} sudo salt '*' state.highstate > $LOG

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

  local kube_cert="kubecfg.crt"
  local kube_key="kubecfg.key"
  local ca_cert="kubernetes.ca.crt"
  # TODO use token instead of kube_auth
  local kube_auth="kubernetes_auth"

  local kubectl="${KUBE_ROOT}/cluster/kubectl.sh"
  local context="${INSTANCE_PREFIX}"
  local user="${INSTANCE_PREFIX}-admin"
  local config_dir="${HOME}/.kube/${context}"

  # TODO: generate ADMIN (and KUBELET) tokens and put those in the master's
  # config file.  Distribute the same way the htpasswd is done.
  (
    mkdir -p "${config_dir}"
    umask 077
    ssh -oStrictHostKeyChecking=no -i ${AWS_SSH_KEY} ubuntu@${KUBE_MASTER_IP} sudo cat /srv/kubernetes/kubecfg.crt >"${config_dir}/${kube_cert}" 2>$LOG
    ssh -oStrictHostKeyChecking=no -i ${AWS_SSH_KEY} ubuntu@${KUBE_MASTER_IP} sudo cat /srv/kubernetes/kubecfg.key >"${config_dir}/${kube_key}" 2>$LOG
    ssh -oStrictHostKeyChecking=no -i ${AWS_SSH_KEY} ubuntu@${KUBE_MASTER_IP} sudo cat /srv/kubernetes/ca.crt >"${config_dir}/${ca_cert}" 2>$LOG

    "${kubectl}" config set-cluster "${context}" --server="https://${KUBE_MASTER_IP}" --certificate-authority="${config_dir}/${ca_cert}" --global
    "${kubectl}" config set-credentials "${user}" --auth-path="${config_dir}/${kube_auth}" --global
    "${kubectl}" config set-context "${context}" --cluster="${context}" --user="${user}" --global
    "${kubectl}" config use-context "${context}" --global

    cat << EOF > "${config_dir}/${kube_auth}"
{
  "User": "$KUBE_USER",
  "Password": "$KUBE_PASSWORD",
  "CAFile": "${config_dir}/${ca_cert}",
  "CertFile": "${config_dir}/${kube_cert}",
  "KeyFile": "${config_dir}/${kube_key}"
}
EOF

    chmod 0600 "${config_dir}/${kube_auth}" "${config_dir}/$kube_cert" \
      "${config_dir}/${kube_key}" "${config_dir}/${ca_cert}"
    echo "Wrote ${config_dir}/${kube_auth}"
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
        local output=$(ssh -oStrictHostKeyChecking=no -i ${AWS_SSH_KEY} ubuntu@$minion_ip sudo docker ps -a 2>/dev/null)
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
        ssh -oStrictHostKeyChecking=no -i ${AWS_SSH_KEY} ubuntu@$$minion_ip sudo service docker start > $LOG 2>&1
        attempt=$(($attempt+1))
        sleep 30
      done
  done

  echo
  echo -e "${color_green}Kubernetes cluster is running.  The master is running at:"
  echo
  echo -e "${color_yellow}  https://${KUBE_MASTER_IP}"
  echo
  echo -e "${color_green}The user name and password to use is located in ${config_dir}/${kube_auth}.${color_norm}"
  echo
}

function kube-down {
  instance_ids=$($AWS_CMD describe-instances | get_instance_ids)
  if [[ -n ${instance_ids} ]]; then
    $AWS_CMD terminate-instances --instance-ids $instance_ids > $LOG
    echo "Waiting for instances deleted"
    while true; do
      instance_states=$($AWS_CMD describe-instances --instance-ids $instance_ids | expect_instance_states terminated)
      if [[ "$instance_states" == "" ]]; then
        echo "All instances terminated"
        break
      else
        echo "Instances not yet terminated: $instance_states"
        echo "Sleeping for 3 seconds..."
        sleep 3
      fi
    done
  fi

  echo "Deleting VPC"
  sec_group_id=$($AWS_CMD describe-security-groups | get_sec_group_id)
  if [[ -n "${sec_group_id}" ]]; then
    $AWS_CMD delete-security-group --group-id $sec_group_id > $LOG
  fi

  vpc_id=$($AWS_CMD describe-vpcs | get_vpc_id)
  if [[ -n "${vpc_id}" ]]; then
    subnet_id=$($AWS_CMD describe-subnets | get_subnet_id $vpc_id)
    igw_id=$($AWS_CMD describe-internet-gateways | get_igw_id $vpc_id)
    route_table_id=$($AWS_CMD describe-route-tables | get_route_table_id $vpc_id)

    $AWS_CMD delete-subnet --subnet-id $subnet_id > $LOG
    $AWS_CMD detach-internet-gateway --internet-gateway-id $igw_id --vpc-id $vpc_id > $LOG
    $AWS_CMD delete-internet-gateway --internet-gateway-id $igw_id > $LOG
    $AWS_CMD delete-route --route-table-id $route_table_id --destination-cidr-block 0.0.0.0/0 > $LOG

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
    echo "readonly SERVER_BINARY_TAR_URL='https://s3-${ZONE}.amazonaws.com/${SERVER_BINARY_TAR_URL}'"
    echo "readonly SALT_TAR_URL='https://s3-${ZONE}.amazonaws.com/${SALT_TAR_URL}'"
    grep -v "^#" "${KUBE_ROOT}/cluster/aws/templates/common.sh"
    grep -v "^#" "${KUBE_ROOT}/cluster/aws/templates/download-release.sh"
    echo "echo Executing configuration"
    echo "sudo salt '*' mine.update"
    echo "sudo salt --force-color '*' state.highstate"
  ) | ssh -oStrictHostKeyChecking=no -i ${AWS_SSH_KEY} ubuntu@${KUBE_MASTER_IP} sudo bash

  get-password

  echo
  echo "Kubernetes cluster is running.  The master is running at:"
  echo
  echo "  https://${KUBE_MASTER_IP}"
  echo

}

function setup-logging-firewall {
  echo "TODO: setup logging"
}

function teardown-logging-firewall {
  echo "TODO: teardown logging"
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
  if [ -z "ip" ]; then
    echo "Could not detect IP for ${node}."
    exit 1
  fi

  for try in $(seq 1 5); do
    if ssh -oLogLevel=quiet -oStrictHostKeyChecking=no -i ${AWS_SSH_KEY} ubuntu@${ip} "${cmd}"; then
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

# Setup monitoring firewalls using heapster and InfluxDB
function setup-monitoring-firewall {
  if [[ "${ENABLE_CLUSTER_MONITORING}" != "true" ]]; then
    return
  fi

  # TODO: Support monitoring firewall
  echo "Cluster monitoring setup is not (yet) supported on AWS"
}

function teardown-monitoring-firewall {
  if [[ "${ENABLE_CLUSTER_MONITORING}" != "true" ]]; then
    return
  fi

  # TODO: Support monitoring firewall
}

function setup-logging-firewall {
  # If logging with Fluentd to Elasticsearch is enabled then create pods
  # and services for Elasticsearch (for ingesting logs) and Kibana (for
  # viewing logs).
  if [[ "${ENABLE_NODE_LOGGING-}" != "true" ]] || \
     [[ "${LOGGING_DESTINATION-}" != "elasticsearch" ]] || \
     [[ "${ENABLE_CLUSTER_LOGGING-}" != "true" ]]; then
    return
  fi

  # TODO: Support logging
  echo "Logging setup is not (yet) supported on AWS"

#  detect-project
#  gcloud compute firewall-rules create "${INSTANCE_PREFIX}-fluentd-elasticsearch-logging" --project "${PROJECT}" \
#    --allow tcp:5601 tcp:9200 tcp:9300 --target-tags "${MINION_TAG}" --network="${NETWORK}"
#
#  # This should be nearly instant once kube-addons gets a chance to
#  # run, and we already know we can hit the apiserver, but it's still
#  # worth checking.
#  echo "waiting for logging services to be created by the master."
#  local kubectl="${KUBE_ROOT}/cluster/kubectl.sh"
#  for i in `seq 1 10`; do
#    if "${kubectl}" get services -l name=kibana-logging -o template -t {{range.items}}{{.id}}{{end}} | grep -q kibana-logging &&
#      "${kubectl}" get services -l name=elasticsearch-logging -o template -t {{range.items}}{{.id}}{{end}} | grep -q elasticsearch-logging; then
#      break
#    fi
#    sleep 10
#  done
#
#  local -r region="${ZONE::-2}"
#  local -r es_ip=$(gcloud compute forwarding-rules --project "${PROJECT}" describe --region "${region}" elasticsearch-logging | grep IPAddress | awk '{print $2}')
#  local -r kibana_ip=$(gcloud compute forwarding-rules --project "${PROJECT}" describe --region "${region}" kibana-logging | grep IPAddress | awk '{print $2}')
#  echo
#  echo -e "${color_green}Cluster logs are ingested into Elasticsearch running at ${color_yellow}http://${es_ip}:9200"
#  echo -e "${color_green}Kibana logging dashboard will be available at ${color_yellow}http://${kibana_ip}:5601${color_norm}"
#  echo
}

function teardown-logging-firewall {
  if [[ "${ENABLE_NODE_LOGGING-}" != "true" ]] || \
     [[ "${LOGGING_DESTINATION-}" != "elasticsearch" ]] || \
     [[ "${ENABLE_CLUSTER_LOGGING-}" != "true" ]]; then
    return
  fi

  # TODO: Support logging
}

# Perform preparations required to run e2e tests
function prepare-e2e() {
  # (AWS runs detect-project, I don't think we need to anything)
  # Note: we can't print anything here, or else the test tools will break with the extra output
  return
}
