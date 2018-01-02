#!/bin/bash
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
if [ "$1" == "" ]; then
  echo "Usage: $0 <config.sh file>"
  exit 1;
fi
source ${DIR}/config.sh $1
source ${DIR}/util.sh

GCLOUD="gcloud"

Header "Deleting prometheus instances..."
for i in `seq 0 $((${PROMETHEUS_NUM_REPLICAS} - 1))`; do
  echo "Deleting instance ${PROMETHEUS_MACHINES[${i}]}..."
  set +e
  ${GCLOUD} compute instances delete -q ${PROMETHEUS_MACHINES[${i}]} \
      --zone ${PROMETHEUS_ZONES[${i}]} \
      --delete-disks all &
  set -e
done
wait

for i in `seq 0 $((${PROMETHEUS_NUM_REPLICAS} - 1))`; do
  echo "Deleting disk ${PROMETHEUS_DISKS[${i}]}..."
  set +e
  ${GCLOUD} compute disks delete -q ${PROMETHEUS_DISKS[${i}]} \
      --zone ${PROMETHEUS_ZONES[${i}]} > /dev/null &
  set -e
done
wait


