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

# Find the release to use.  If passed in, go with that and validate.  If not use
# the release/config.sh version assuming a dev workflow.
function find-release() {
  if [ -n "$1" ]; then
    RELEASE_NORMALIZED=$1
  else
    local RELEASE_CONFIG_SCRIPT=$(dirname $0)/../../release/aws/config.sh
    if [ -f $(dirname $0)/../../release/aws/config.sh ]; then
      . $RELEASE_CONFIG_SCRIPT
      normalize_release
    fi
  fi

  # Do one final check that we have a good release
  if ! aws s3 ls $RELEASE_NORMALIZED/$RELEASE_TAR_FILE | grep $RELEASE_TAR_FILE > /dev/null; then
    echo "Could not find release tar.  If developing, make sure you have run src/release/release.sh to create a release."
    exit 1
  fi
  echo "Release: ${RELEASE_NORMALIZED}"
}

function json_val {
    python -c 'import json,sys;obj=json.load(sys.stdin);print obj'$1''
}

function get_instance_ids {
  python -c 'import json,sys; lst = [str(instance["InstanceId"]) for reservation in json.load(sys.stdin)["Reservations"] for instance in reservation["Instances"] for tag in instance["Tags"] if tag["Value"].startswith("kubernetes-minion") or tag["Value"].startswith("kubernetes-master")]; print " ".join(lst)'
}

function get_vpc_id {
  python -c 'import json,sys; lst = [str(vpc["VpcId"]) for vpc in json.load(sys.stdin)["Vpcs"] for tag in vpc["Tags"] if tag["Value"] == "kubernetes-vpc"]; print "".join(lst)'
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

function get_instance_states {
  python -c 'import json,sys; lst = [str(instance["InstanceId"]) for reservation in json.load(sys.stdin)["Reservations"] for instance in reservation["Instances"] if instance["State"]["Name"] != "terminated"]; print " ".join(lst)'
}

function detect-master {
  echo
}

function detect-minions {
  echo
}

function get-password {
  file=${HOME}/.kubernetes_auth
  if [ -e ${file} ]; then
    user=$(cat $file | python -c 'import json,sys;print json.load(sys.stdin)["User"]')
    passwd=$(cat $file | python -c 'import json,sys;print json.load(sys.stdin)["Password"]')
    return
  fi
  user=admin
  passwd=$(python -c 'import string,random; print "".join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(16))')

  # Store password for reuse.
  cat << EOF > ~/.kubernetes_auth
{
  "User": "$user",
  "Password": "$passwd"
}
EOF
  chmod 0600 ~/.kubernetes_auth
}

# Verify prereqs
function verify-prereqs {
  if [ "$(which aws)" == "" ]; then
    echo "Can't find aws in PATH, please fix and retry."
    exit 1
  fi
}

function kube-up {

    # Find the release to use.  Generally it will be passed when doing a 'prod'
    # install and will default to the release/config.sh version when doing a
    # developer up.
    find-release $1

    # Build up start up script for master
    KUBE_TEMP=$(mktemp -d -t kubernetes.XXXXXX)
    trap "rm -rf ${KUBE_TEMP}" EXIT

    get-password
    echo "Using password: $user:$passwd"
    python $(dirname $0)/../../third_party/htpasswd/htpasswd.py -b -c ${KUBE_TEMP}/htpasswd $user $passwd
    HTPASSWD=$(cat ${KUBE_TEMP}/htpasswd)

    if [ ! -f $AWS_SSH_KEY ]; then
        ssh-keygen -f $AWS_SSH_KEY -N ''
    fi

    AWS_CMD="aws --output json ec2"
    
    $AWS_CMD import-key-pair --key-name kubernetes --public-key-material file://$AWS_SSH_KEY.pub > /dev/null 2>&1 || true
    VPC_ID=$($AWS_CMD create-vpc --cidr-block 10.244.0.0/16 | json_val '["Vpc"]["VpcId"]')
    $AWS_CMD create-tags --resources $VPC_ID --tags Key=Name,Value=kubernetes-vpc > /dev/null
    SUBNET_ID=$($AWS_CMD create-subnet --cidr-block 10.244.0.0/24 --vpc-id $VPC_ID | json_val '["Subnet"]["SubnetId"]')
    IGW_ID=$($AWS_CMD create-internet-gateway | json_val '["InternetGateway"]["InternetGatewayId"]')
    $AWS_CMD attach-internet-gateway --internet-gateway-id $IGW_ID --vpc-id $VPC_ID > /dev/null
    ROUTE_TABLE_ID=$($AWS_CMD describe-route-tables --filters Name=vpc-id,Values=$VPC_ID | json_val '["RouteTables"][0]["RouteTableId"]')
    $AWS_CMD associate-route-table --route-table-id $ROUTE_TABLE_ID --subnet-id $SUBNET_ID > /dev/null
    $AWS_CMD describe-route-tables --filters Name=vpc-id,Values=$VPC_ID > /dev/null
    $AWS_CMD create-route --route-table-id $ROUTE_TABLE_ID --destination-cidr-block 0.0.0.0/0 --gateway-id $IGW_ID > /dev/null
    SEC_GROUP_ID=$($AWS_CMD create-security-group --group-name kubernetes-sec-group --description kubernetes-sec-group --vpc-id $VPC_ID | json_val '["GroupId"]')
    $AWS_CMD authorize-security-group-ingress --group-id $SEC_GROUP_ID --protocol -1 --port all --cidr 0.0.0.0/0 > /dev/null

    (
    echo "#!/bin/bash"
    echo "MASTER_NAME=${MASTER_NAME}"
    echo "MASTER_RELEASE_TAR=${RELEASE_FULL_HTTP_PATH}/master-release.tgz"
    echo "MASTER_HTPASSWD='${HTPASSWD}'"
    echo "NUM_MINIONS=${NUM_MINIONS}"
    grep -v "^#" $(dirname $0)/templates/download-release.sh
    grep -v "^#" $(dirname $0)/templates/salt-master.sh
    ) > ${KUBE_TEMP}/master-start.sh

    master_id=$($AWS_CMD run-instances \
    --image-id $IMAGE \
    --instance-type $MASTER_SIZE \
    --subnet-id $SUBNET_ID \
    --private-ip-address 10.244.0.251 \
    --key-name kubernetes \
    --security-group-ids $SEC_GROUP_ID \
    --associate-public-ip-address \
    --user-data file://${KUBE_TEMP}/master-start.sh | json_val '["Instances"][0]["InstanceId"]')

    $AWS_CMD create-tags --resources $master_id --tags Key=Name,Value=kubernetes-master > /dev/null

    for i in $(seq 1 $NUM_MINIONS); do
    (
      echo "#! /bin/bash"
      echo "MASTER_NAME=${MASTER_NAME}"
      echo "NUM_MINIONS=${NUM_MINIONS}"
      echo "MINION_IP_RANGE=${MINION_IP_RANGE}"
      grep -v "^#" $(dirname $0)/templates/salt-minion.sh
    ) > ${KUBE_TEMP}/minion-start-${i}.sh

        minion_id=$($AWS_CMD run-instances \
        --image-id $IMAGE \
        --instance-type $MINION_SIZE \
        --subnet-id $SUBNET_ID \
        --private-ip-address 10.244.0.10${i} \
        --key-name kubernetes \
        --security-group-ids $SEC_GROUP_ID \
        --associate-public-ip-address \
        --user-data file://${KUBE_TEMP}/minion-start-${i}.sh | json_val '["Instances"][0]["InstanceId"]')

        $AWS_CMD create-tags --resources $minion_id --tags Key=Name,Value=kubernetes-minion-$i > /dev/null
    done
}

function kube-down {
  AWS_CMD="aws --output json ec2"
  instance_ids=$($AWS_CMD describe-instances | get_instance_ids)
  $AWS_CMD terminate-instances --instance-ids $instance_ids > /dev/null
  echo "Waiting for instances deleted"
  while true; do
    instance_states=$($AWS_CMD describe-instances --instance-ids $instance_ids | get_instance_states)
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

  $AWS_CMD delete-subnet --subnet-id $subnet_id > /dev/null
  $AWS_CMD detach-internet-gateway --internet-gateway-id $igw_id --vpc-id $vpc_id > /dev/null
  $AWS_CMD delete-internet-gateway --internet-gateway-id $igw_id > /dev/null
  $AWS_CMD delete-security-group --group-id $sec_group_id > /dev/null
  $AWS_CMD delete-route --route-table-id $route_table_id --destination-cidr-block 0.0.0.0/0 > /dev/null
  $AWS_CMD delete-vpc --vpc-id $vpc_id > /dev/null
}
