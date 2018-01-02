#!/bin/bash
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
if [ "$1" == "" ]; then
  echo "Usage: $0 <config.sh file>"
  exit 1;
fi
source ${DIR}/config.sh $1
source ${DIR}/util.sh

GCLOUD="gcloud"

Header "Deleting log instances..."
for i in `seq 0 $((${LOG_NUM_REPLICAS} - 1))`; do
  echo "Deleting instance ${LOG_MACHINES[${i}]}..."
  set +e
  ${GCLOUD} compute instances delete -q ${LOG_MACHINES[${i}]} \
      --zone ${LOG_ZONES[${i}]} \
      --delete-disks all &
  set -e
done
wait

for i in `seq 0 $((${LOG_NUM_REPLICAS} - 1))`; do
  echo "Deleting disk ${LOG_DISKS[${i}]}..."
  set +e
  ${GCLOUD} compute disks delete -q ${LOG_DISKS[${i}]} \
      --zone ${LOG_ZONES[${i}]} > /dev/null &
  set -e
done
wait


