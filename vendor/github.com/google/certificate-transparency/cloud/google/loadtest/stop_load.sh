#!/bin/bash
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
if [ "$1" == "" ]; then
  echo "Usage $0: <config-file>"
  exit 1
fi
source ${DIR}/../util.sh
source ${1}

GCLOUD="gcloud"

for i in `seq 0 $((${LOAD_NUM_REPLICAS} - 1))`; do
  echo "Deleting instance ${i}"

  ${GCLOUD} compute instances delete -q load-${i} \
      --zone=${ZONE} \
      --delete-disks all &
done
wait
