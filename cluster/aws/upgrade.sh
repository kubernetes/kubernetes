#!/bin/bash

# Copyright 2015 The Kubernetes Authors All rights reserved.
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

# !!!EXPERIMENTAL !!! Upgrade script for AWS.

set -o errexit
set -o nounset
set -o pipefail

if [[ "${KUBERNETES_PROVIDER:-aws}" != "aws" ]]; then
  echo "!!! ${1} only works on AWS" >&2
  exit 1
fi

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/cluster/kube-util.sh"

KUBERNETES_RELEASE=$(cat ${KUBE_ROOT}/version)
AWS_S3_BUCKET="kubernetes-artifacts-${KUBERNETES_RELEASE}"


AWS_ASG_CMD="aws autoscaling"

create-bootstrap-script
find-release-tars
upload-server-tars
ensure-temp-dir

${AWS_ASG_CMD} describe-launch-configurations \
--launch-configuration-names ${ASG_NAME} \
--query 'LaunchConfigurations[].UserData' |base64 -d|gunzip - > ${KUBE_TEMP}/node-user-data

sed -i.bak "s#SERVER_BINARY_TAR_URL:.*#SERVER_BINARY_TAR_URL: '${SERVER_BINARY_TAR_URL}'#"  ${KUBE_TEMP}/node-user-data
sed -i.bak "s#SERVER_BINARY_TAR_HASH:.*#SERVER_BINARY_TAR_HASH: '${SERVER_BINARY_TAR_HASH}'#"  ${KUBE_TEMP}/node-user-data
sed -i.bak "s#SALT_TAR_URL:.*#SALT_TAR_URL: '${SALT_TAR_URL}'#"  ${KUBE_TEMP}/node-user-data
sed -i.bak "s#SALT_TAR_HASH:.*#SALT_TAR_HASH: '${SALT_TAR_HASH}'#"  ${KUBE_TEMP}/node-user-data
sed -i.bak "s#wget -O bootstrap.*#wget -O bootstrap ${BOOTSTRAP_SCRIPT_URL}#"  ${KUBE_TEMP}/node-user-data


${AWS_ASG_CMD} describe-launch-configurations \
--launch-configuration-names ${ASG_NAME} \
--output json --query "LaunchConfigurations[0]" > ${KUBE_TEMP}/launch-configuration.json

sed -i.bak "/LaunchConfigurationARN/d" ${KUBE_TEMP}/launch-configuration.json
sed -i.bak "/CreatedTime/d" ${KUBE_TEMP}/launch-configuration.json
sed -i.bak "/KernelId/d" ${KUBE_TEMP}/launch-configuration.json
sed -i.bak "/RamdiskId/d" ${KUBE_TEMP}/launch-configuration.json
sed -i.bak "/LaunchConfigurationName/d" ${KUBE_TEMP}/launch-configuration.json
sed -i.bak "/UserData/d" ${KUBE_TEMP}/launch-configuration.json


gzip ${KUBE_TEMP}/node-user-data

echo "Create Launch Configuration ${ASG_NAME}-${KUBERNETES_RELEASE} for Kubernetes ${KUBERNETES_RELEASE}"
${AWS_ASG_CMD} create-launch-configuration \
    --launch-configuration-name ${ASG_NAME}-${KUBERNETES_RELEASE} \
    --cli-input-json file://${KUBE_TEMP}/launch-configuration.json \
    --user-data fileb://${KUBE_TEMP}/node-user-data.gz

echo "Update Auto Scaling Group ${ASG_NAME} to use Launch Configuration ${ASG_NAME}-${KUBERNETES_RELEASE}"
${AWS_ASG_CMD} update-auto-scaling-group \
--auto-scaling-group-name ${ASG_NAME} \
--launch-configuration-name ${ASG_NAME}-${KUBERNETES_RELEASE}

echo "Please delete Minon nodes create from Launch Configuration ${ASG_NAME}"
