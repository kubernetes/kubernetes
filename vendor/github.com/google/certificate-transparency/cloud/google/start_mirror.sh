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

Header "Creating mirror persistent disks..."
for i in `seq 0 $((${MIRROR_NUM_REPLICAS} - 1))`; do
  echo "Creating disk ${MIRROR_DISKS[${i}]}..."
  ${GCLOUD} compute disks create -q ${MIRROR_DISKS[${i}]} \
      --zone ${MIRROR_ZONES[${i}]} \
      --size=${MIRROR_DISK_SIZE} &
done
wait

for i in `seq 0 $((${MIRROR_NUM_REPLICAS} - 1))`; do
  echo "Waiting for disk ${MIRROR_DISKS[${i}]}..."
  WaitForStatus disks ${MIRROR_DISKS[${i}]} ${MIRROR_ZONES[${i}]} READY &
done
wait

MANIFEST=/tmp/mirror_container.yaml

Header "Creating mirror instances..."
for i in `seq 0 $((${MIRROR_NUM_REPLICAS} - 1))`; do
  echo "Creating instance ${MIRROR_MACHINES[$i]}"

  echo "${MIRROR_META[${i}]}" > ${MANIFEST}.${i}

  ${GCLOUD} compute instances create -q ${MIRROR_MACHINES[${i}]} \
      --zone ${MIRROR_ZONES[${i}]} \
      --machine-type ${MIRROR_MACHINE_TYPE} \
      --image container-vm \
      --disk name=${MIRROR_DISKS[${i}]},mode=rw,boot=no,auto-delete=yes \
      --tags mirror-node \
      --scopes "monitoring,storage-ro,compute-ro,logging-write" \
      --metadata-from-file startup-script=${DIR}/node_init.sh,google-container-manifest=${MANIFEST}.${i} &
done
wait

for i in `seq 0 $((${MIRROR_NUM_REPLICAS} - 1))`; do
  echo "Waiting for instance ${MIRROR_MACHINES[${i}]}..."
  WaitForStatus instances ${MIRROR_MACHINES[${i}]} ${MIRROR_ZONES[${i}]} RUNNING &
done
wait


