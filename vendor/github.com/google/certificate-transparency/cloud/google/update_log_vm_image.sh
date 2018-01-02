#!/bin/bash
# TODO(alcutter): Factor out common code with update_mirror_vm_image.sh script
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

MANIFEST=/tmp/log_container.yaml

Header "Recreating log instances..."
for i in `seq 0 $((${LOG_NUM_REPLICAS} - 1))`; do
  echo "Deleting instance ${LOG_MACHINES[$i]}"
  set +e
   ${GCLOUD} compute instances delete -q ${LOG_MACHINES[${i}]} \
      --zone ${LOG_ZONES[${i}]} \
      --keep-disks data
  set -e

  echo "${LOG_META[${i}]}" > ${MANIFEST}.${i}

  echo "Recreating instance ${LOG_MACHINES[$i]}"
  ${GCLOUD} compute instances create -q ${LOG_MACHINES[${i}]} \
      --zone ${LOG_ZONES[${i}]} \
      --machine-type ${LOG_MACHINE_TYPE} \
      --image container-vm \
      --disk name=${LOG_DISKS[${i}]},mode=rw,boot=no,auto-delete=no \
      --tags log-node \
      --scopes "monitoring,storage-ro,compute-ro,logging-write" \
      --metadata-from-file startup-script=${DIR}/node_init.sh,google-container-manifest=${MANIFEST}.${i}

  gcloud compute instance-groups unmanaged add-instances \
      "log-group-${LOG_ZONES[${i}]}" \
      --zone ${LOG_ZONES[${i}]} \
      --instances ${LOG_MACHINES[${i}]} &

  set +e
  echo "Waiting for instance ${LOG_MACHINES[${i}]}..."
  WaitForStatus instances ${LOG_MACHINES[${i}]} ${LOG_ZONES[${i}]} RUNNING
  echo "Waiting for log service on ${LOG_MACHINES[${i}]}..."
  WaitHttpStatus ${LOG_MACHINES[${i}]} ${LOG_ZONES[${i}]} /ct/v1/get-sth 200
  set -e
done
