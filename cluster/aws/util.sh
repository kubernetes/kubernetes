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
source $(dirname ${BASH_SOURCE})/${KUBE_CONFIG_FILE-"config-default.sh"}

AWS_CMD="aws --output json ec2"

function json_val {
    python -c 'import json,sys;obj=json.load(sys.stdin);print obj'$1''
}

# TODO (ayurchuk) Refactor the get_* functions to use filters
# TODO (bburns) Parameterize this for multiple cluster per project
function get_instance_ids {
  python -c 'import json,sys; lst = [str(instance["InstanceId"]) for reservation in json.load(sys.stdin)["Reservations"] for instance in reservation["Instances"] for tag in instance.get("Tags", []) if tag["Value"].startswith("kubernetes-minion") or tag["Value"].startswith("kubernetes-master")]; print " ".join(lst)'
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
  python -c "import json,sys; lst = [str(instance['NetworkInterfaces'][0]['Association']['PublicIp']) for reservation in json.load(sys.stdin)['Reservations'] for instance in reservation['Instances'] for tag in instance.get('Tags', []) if tag['Value'] == '$1' and instance['State']['Name'] == 'running']; print ' '.join(lst)"
}

function detect-master () {
  KUBE_MASTER=${MASTER_NAME}
  if [[ -z "${KUBE_MASTER_IP-}" ]]; then
    KUBE_MASTER_IP=$($AWS_CMD describe-instances | get_instance_public_ip $MASTER_NAME)
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
    local minion_ip=$($AWS_CMD describe-instances --filters Name=tag-value,Values=${MINION_NAMES[$i]} Name=instance-state-name,Values=running | get_instance_public_ip ${MINION_NAMES[$i]})
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

function setup-monitoring {
  if [[ "${ENABLE_CLUSTER_MONITORING:-false}" == "true" ]]; then
    # TODO: Implement this.
    echo "Monitoring not currently supported on AWS"
  fi
}

function teardown-monitoring {
  if [[ "${ENABLE_CLUSTER_MONITORING:-false}" == "true" ]]; then
    # TODO: Implement this.
    echo "Monitoring not currently supported on AWS"
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

  $AWS_CMD import-key-pair --key-name kubernetes --public-key-material file://$AWS_SSH_KEY.pub > $LOG 2>&1 || true

  VPC_ID=$($AWS_CMD describe-vpcs | get_vpc_id)

  if [ -z "$VPC_ID" ]; then
	  echo "Creating vpc."
	  VPC_ID=$($AWS_CMD create-vpc --cidr-block 172.20.0.0/16 | json_val '["Vpc"]["VpcId"]')
	  $AWS_CMD modify-vpc-attribute --vpc-id $VPC_ID --enable-dns-support '{"Value": true}' > $LOG
	  $AWS_CMD modify-vpc-attribute --vpc-id $VPC_ID --enable-dns-hostnames '{"Value": true}' > $LOG
	  $AWS_CMD create-tags --resources $VPC_ID --tags Key=Name,Value=kubernetes-vpc > $LOG

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
    echo "readonly MASTER_NAME='${MASTER_NAME}'"
    echo "readonly NODE_INSTANCE_PREFIX='${INSTANCE_PREFIX}-minion'"
    echo "readonly SERVER_BINARY_TAR_URL='https://s3.amazonaws.com/${SERVER_BINARY_TAR_URL}'"
    echo "readonly SALT_TAR_URL='https://s3.amazonaws.com/${SALT_TAR_URL}'"
    echo "readonly AWS_ZONE='${ZONE}'"
    echo "readonly MASTER_HTPASSWD='${htpasswd}'"
    echo "readonly PORTAL_NET='${PORTAL_NET}'"
    echo "readonly ENABLE_NODE_MONITORING='${ENABLE_NODE_MONITORING:-false}'"
    echo "readonly ENABLE_NODE_LOGGING='${ENABLE_NODE_LOGGING:-false}'"
    echo "readonly LOGGING_DESTINATION='${LOGGING_DESTINATION:-}'"
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
  sleep 3
  $AWS_CMD create-tags --resources $master_id --tags Key=Name,Value=$MASTER_NAME > $LOG
  sleep 3
  $AWS_CMD create-tags --resources $master_id --tags Key=Role,Value=$MASTER_TAG > $LOG

  for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
    echo "Starting Minion (${MINION_NAMES[$i]})"
    (
      # We pipe this to the ami as a startup script in the user-data field.  Requires a compatible ami
      echo "#! /bin/bash"
      echo "MASTER_NAME='${MASTER_NAME}'"
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
    sleep 3
    n=0
    until [ $n -ge 5 ]; do
      $AWS_CMD create-tags --resources $minion_id --tags Key=Name,Value=${MINION_NAMES[$i]} > $LOG && break
      n=$[$n+1]
      sleep 15
    done

    sleep 3
    n=0
    until [ $n -ge 5 ]; do
      $AWS_CMD create-tags --resources $minion_id --tags Key=Role,Value=$MINION_TAG > $LOG && break
      n=$[$n+1]
      sleep 15
    done

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
  echo "Waiting for cluster to settle"
  local i
  for (( i=0; i < 6*3; i++)); do
    printf "."
    sleep 10
  done
  echo "Re-running salt highstate"
  ssh -oStrictHostKeyChecking=no -i ~/.ssh/kube_aws_rsa ubuntu@${KUBE_MASTER_IP} sudo salt '*' state.highstate > $LOG

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
  echo "Sanity checking cluster..."

  sleep 5

  # Don't bail on errors, we want to be able to print some info.
  set +e

  # Basic sanity checking
  for i in ${KUBE_MINION_IP_ADDRESSES[@]}; do
    # Make sure docker is installed
    ssh -oStrictHostKeyChecking=no ubuntu@$i -i ~/.ssh/kube_aws_rsa which docker > $LOG 2>&1
    if [ "$?" != "0" ]; then
      echo "Docker failed to install on $i. Your cluster is unlikely to work correctly."
      echo "Please run ./cluster/aws/kube-down.sh and re-create the cluster. (sorry!)"
      exit 1
    fi
  done

  echo
  echo "Kubernetes cluster is running.  Access the master at:"
  echo
  echo "  https://${KUBE_USER}:${KUBE_PASSWORD}@${KUBE_MASTER_IP}"
  echo

  local kube_cert=".kubecfg.crt"
  local kube_key=".kubecfg.key"
  local ca_cert=".kubernetes.ca.crt"

  # TODO: generate ADMIN (and KUBELET) tokens and put those in the master's
  # config file.  Distribute the same way the htpasswd is done.
  (
    umask 077
    ssh -oStrictHostKeyChecking=no -i ~/.ssh/kube_aws_rsa ubuntu@${KUBE_MASTER_IP} sudo cat /srv/kubernetes/kubecfg.crt >"${HOME}/${kube_cert}" 2>$LOG
    ssh -oStrictHostKeyChecking=no -i ~/.ssh/kube_aws_rsa ubuntu@${KUBE_MASTER_IP} sudo cat /srv/kubernetes/kubecfg.key >"${HOME}/${kube_key}" 2>$LOG
    ssh -oStrictHostKeyChecking=no -i ~/.ssh/kube_aws_rsa ubuntu@${KUBE_MASTER_IP} sudo cat /srv/kubernetes/ca.crt >"${HOME}/${ca_cert}" 2>$LOG

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

function kube-down {
  AWS_CMD="aws --output json ec2"
  instance_ids=$($AWS_CMD describe-instances | get_instance_ids)
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

  echo "Deleting VPC"
  vpc_id=$($AWS_CMD describe-vpcs | get_vpc_id)
  subnet_id=$($AWS_CMD describe-subnets | get_subnet_id $vpc_id)
  igw_id=$($AWS_CMD describe-internet-gateways | get_igw_id $vpc_id)
  route_table_id=$($AWS_CMD describe-route-tables | get_route_table_id $vpc_id)
  sec_group_id=$($AWS_CMD describe-security-groups | get_sec_group_id)

  $AWS_CMD delete-subnet --subnet-id $subnet_id > $LOG
  $AWS_CMD detach-internet-gateway --internet-gateway-id $igw_id --vpc-id $vpc_id > $LOG
  $AWS_CMD delete-internet-gateway --internet-gateway-id $igw_id > $LOG
  $AWS_CMD delete-security-group --group-id $sec_group_id > $LOG
  $AWS_CMD delete-route --route-table-id $route_table_id --destination-cidr-block 0.0.0.0/0 > $LOG
  $AWS_CMD delete-vpc --vpc-id $vpc_id > $LOG
}
