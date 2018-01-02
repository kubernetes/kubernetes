#!/bin/bash
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
if [ "$1" == "" ]; then
  echo "Usage: $0 <config.sh file>"
  exit 1;
fi
source ${DIR}/config.sh $1
source ${DIR}/util.sh

GCLOUD="gcloud"

Header "Deleting mirror instances..."
for i in `seq 0 $((${MIRROR_NUM_REPLICAS} - 1))`; do
  echo "Deleting instance ${MIRROR_MACHINES[${i}]}..."
  set +e
  ${GCLOUD} compute instances delete -q ${MIRROR_MACHINES[${i}]} \
      --zone ${MIRROR_ZONES[${i}]} \
      --delete-disks all &
  set -e
done
wait

for i in `seq 0 $((${MIRROR_NUM_REPLICAS} - 1))`; do
  echo "Deleting disk ${MIRROR_DISKS[${i}]}..."
  set +e
  ${GCLOUD} compute disks delete -q ${MIRROR_DISKS[${i}]} \
      --zone ${MIRROR_ZONES[${i}]} > /dev/null &
  set -e
done
wait


