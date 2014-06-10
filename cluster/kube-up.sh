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

# Bring up a Kubernetes cluster.
#
# If the full release name (gs://<bucket>/<release>) is passed in then we take
# that directly.  If not then we assume we are doing development stuff and take
# the defaults in the release config.

# exit on any error
set -e

source $(dirname $0)/util.sh

# Make sure that prerequisites are installed.
for x in gcloud gsutil; do
  if [ "$(which $x)" == "" ]; then
    echo "Can't find $x in PATH, please fix and retry."
    exit 1
  fi
done

# Find the release to use.  Generally it will be passed when doing a 'prod'
# install and will default to the release/config.sh version when doing a
# developer up.
find-release $1

# Detect the project into $PROJECT if it isn't set
detect-project

# Build up start up script for master
KUBE_TEMP=$(mktemp -d -t kubernetes.XXXXXX)
trap "rm -rf ${KUBE_TEMP}" EXIT

get-password
echo "Generating password: $user:$passwd"
htpasswd -b -c ${KUBE_TEMP}/htpasswd $user $passwd
cat << EOF > ~/.kubernetes_auth
{
  "User": "$user",
  "Password": "$passwd"
}
EOF
chmod 0600 ~/.kubernetes_auth
HTPASSWD=$(cat ${KUBE_TEMP}/htpasswd)

(
  echo "#! /bin/bash"
  echo "MASTER_NAME=${MASTER_NAME}"
  echo "MASTER_RELEASE_TAR=${RELEASE_NORMALIZED}/master-release.tgz"
  echo "MASTER_HTPASSWD='${HTPASSWD}'"
  grep -v "^#" $(dirname $0)/templates/download-release.sh
  grep -v "^#" $(dirname $0)/templates/salt-master.sh
) > ${KUBE_TEMP}/master-start.sh

echo "Starting VMs and configuring firewalls"
gcloud compute firewalls create --quiet ${MASTER_NAME}-https \
  --project ${PROJECT} \
  --target-tags ${MASTER_TAG} \
  --allow tcp:443 &

gcloud compute instances create ${MASTER_NAME}\
  --project ${PROJECT} \
  --zone ${ZONE} \
  --machine-type ${MASTER_SIZE} \
  --image ${IMAGE} \
  --tags ${MASTER_TAG} \
  --no-scopes \
  --restart-on-failure \
  --metadata-from-file startup-script=${KUBE_TEMP}/master-start.sh &

GCLOUD_VERSION=$(gcloud version | grep compute | cut -f 2 -d ' ')

for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
  (
    echo "#! /bin/bash"
    echo "MASTER_NAME=${MASTER_NAME}"
    echo "MINION_IP_RANGE=${MINION_IP_RANGES[$i]}"
    grep -v "^#" $(dirname $0)/templates/salt-minion.sh
  ) > ${KUBE_TEMP}/minion-start-${i}.sh

  gcloud compute instances create ${MINION_NAMES[$i]} \
    --project ${PROJECT} \
    --zone ${ZONE} \
    --machine-type ${MINION_SIZE} \
    --image ${IMAGE} \
    --tags ${MINION_TAG} \
    --no-scopes \
    --restart-on-failure \
    --can-ip-forward \
    --metadata-from-file startup-script=${KUBE_TEMP}/minion-start-${i}.sh &

  # 'gcloud compute' past 2014.06.08 breaks the way we are specifying
  # --next-hop-instance and there is no way to be compatible with both versions.
  if [[ $GCLOUD_VERSION < "2014.06.08" ]]; then
    gcloud compute routes create ${MINION_NAMES[$i]} \
      --project ${PROJECT} \
      --destination-range ${MINION_IP_RANGES[$i]} \
      --next-hop-instance ${ZONE}/instances/${MINION_NAMES[$i]} &
  else
    gcloud compute routes create ${MINION_NAMES[$i]} \
      --project ${PROJECT} \
      --destination-range ${MINION_IP_RANGES[$i]} \
      --next-hop-instance ${MINION_NAMES[$i]} \
      --next-hop-instance-zone ${ZONE} &
  fi
done

FAIL=0
for job in `jobs -p`
do
    wait $job || let "FAIL+=1"
done
if (( $FAIL != 0 )); then
  echo "${FAIL} commands failed.  Exiting."
  exit 2
fi


detect-master > /dev/null

echo "Waiting for cluster initialization."
echo
echo "  This will continually check to see if the API for kubernetes is reachable."
echo "  This might loop forever if there was some uncaught error during start"
echo "  up."
echo

until $(curl --insecure --user ${user}:${passwd} --max-time 1 \
        --fail --output /dev/null --silent https://${KUBE_MASTER_IP}/api/v1beta1/pods); do
    printf "."
    sleep 2
done

echo
echo "Kubernetes cluster is running.  Access the master at:"
echo
echo "  https://${user}:${passwd}@${KUBE_MASTER_IP}"
echo
echo "Security note: The server above uses a self signed certificate.  This is"
echo "    subject to \"Man in the middle\" type attacks."


