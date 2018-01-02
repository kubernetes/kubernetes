#!/bin/bash
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
if [ "$1" == "" ]; then
  echo "Usage $0: <config-file>"
  exit 1
fi
source ${DIR}/util.sh
source ${DIR}/config.sh $1

GCLOUD="gcloud"

Header "Updating mirror instances..."
i=0
while [ $i -lt ${MIRROR_NUM_REPLICAS} ]; do
  echo "Updating ${MIRROR_MACHINES[${i}]}"
  echo "${MIRROR_META[${i}]}" > /tmp/metadata.${i}

  if ! gcloud compute instances add-metadata \
      ${MIRROR_MACHINES[${i}]} \
      --zone ${MIRROR_ZONES[${i}]} \
      --metadata-from-file google-container-manifest=/tmp/metadata.${i}; then
    echo "Retrying"
    continue
  fi
  

  if ! gcloud compute ssh ${MIRROR_MACHINES[${i}]} \
      --zone ${MIRROR_ZONES[${i}]} \
      --command \
          'sudo docker pull gcr.io/'${PROJECT}'/super_mirror:test &&
           sudo docker kill $(sudo docker ps | grep super_mirror | awk -- "{print \$1}" )'; then
    echo "Retrying"
    continue
  fi

  WaitHttpStatus ${MIRROR_MACHINES[${i}]} ${MIRROR_ZONES[${i}]} /ct/v1/get-sth 200
  i=$(($i + 1))
done;


