#!/bin/bash

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

source "${KUBE_ROOT}/cluster/kubemark/config-default.sh"
source "${KUBE_ROOT}/cluster/kubemark/util.sh"
source "${KUBE_ROOT}/cluster/lib/util.sh"

# hack/lib/init.sh will ovewrite ETCD_VERSION if this is unset
# what what is default in hack/lib/etcd.sh
# To avoid it, if it is empty, we set it to 'avoid-overwrite' and
# clean it after that.
if [ -z "${ETCD_IMAGE}" ]; then
  ETCD_IMAGE="avoid-overwrite"
fi
source "${KUBE_ROOT}/hack/lib/init.sh"
if [ "${ETCD_IMAGE}" == "avoid-overwrite" ]; then
  ETCD_IMAGE=""
fi

detect-project &> /dev/null
export PROJECT
find-release-tars

MASTER_NAME="${INSTANCE_PREFIX}-kubemark-master"
MASTER_TAG="kubemark-master"
EVENT_STORE_NAME="${INSTANCE_PREFIX}-event-store"

RETRIES=3

export KUBECTL="${KUBE_ROOT}/cluster/kubectl.sh"
export KUBEMARK_DIRECTORY="${KUBE_ROOT}/test/kubemark"
export RESOURCE_DIRECTORY="${KUBEMARK_DIRECTORY}/resources"

# Runs gcloud compute command with the given parameters. Up to $RETRIES will be made
# to execute the command.
# arguments:
# $@: all stuff that goes after 'gcloud compute '
function run-gcloud-compute-with-retries {
  for attempt in $(seq 1 ${RETRIES}); do
    if ! gcloud compute $@; then
      echo -e "${color_yellow}Attempt $(($attempt+1)) failed to $1 $2 $3. Retrying.${color_norm}" >& 2
      sleep $(($attempt * 5))
    else
      return 0
    fi
  done
  echo -e "${color_red} Failed to $1 $2 $3.${color_norm}" >& 2
  exit 1
}
