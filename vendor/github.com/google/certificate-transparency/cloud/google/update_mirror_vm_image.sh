#!/bin/bash
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
if [ "$1" == "" ]; then
  echo "Usage: $0 <config.sh file>"
  exit 1;
fi
source ${DIR}/config.sh $1
source ${DIR}/util.sh

set -e
GCLOUD="gcloud"

${GCLOUD} config set project ${PROJECT}

MANIFEST=/tmp/mirror_container.yaml

Header "Recreating mirror instances..."
for i in `seq 0 $((${MIRROR_NUM_REPLICAS} - 1))`; do
  echo "Deleting instance ${MIRROR_MACHINES[$i]}"
  set +e
   ${GCLOUD} compute instances delete -q ${MIRROR_MACHINES[${i}]} \
      --zone ${MIRROR_ZONES[${i}]} \
      --keep-disks data
  set -e

  echo "${MIRROR_META[${i}]}" > ${MANIFEST}.${i}

  echo "Recreating instance ${MIRROR_MACHINES[$i]}"
  ${GCLOUD} compute instances create -q ${MIRROR_MACHINES[${i}]} \
      --zone ${MIRROR_ZONES[${i}]} \
      --machine-type ${MIRROR_MACHINE_TYPE} \
      --image container-vm \
      --disk name=${MIRROR_DISKS[${i}]},mode=rw,boot=no,auto-delete=no \
      --tags mirror-node \
      --scopes "monitoring,storage-ro,compute-ro,logging-write" \
      --metadata-from-file startup-script=${DIR}/node_init.sh,google-container-manifest=${MANIFEST}.${i}

  gcloud compute instance-groups unmanaged add-instances \
      "mirror-group-${MIRROR_ZONES[${i}]}" \
      --zone ${MIRROR_ZONES[${i}]} \
      --instances ${MIRROR_MACHINES[${i}]} &

  set +e
  echo "Waiting for instance ${MIRROR_MACHINES[${i}]}..."
  WaitForStatus instances ${MIRROR_MACHINES[${i}]} ${MIRROR_ZONES[${i}]} RUNNING
  echo "Waiting for mirror service on ${MIRROR_MACHINES[${i}]}..."
  WaitHttpStatus ${MIRROR_MACHINES[${i}]} ${MIRROR_ZONES[${i}]} /ct/v1/get-sth 200
  set -e
done
