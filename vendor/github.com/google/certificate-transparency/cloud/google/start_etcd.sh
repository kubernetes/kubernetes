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

Header "Creating etcd persistent disks..."
for i in `seq 0 $((${ETCD_NUM_REPLICAS} - 1))`; do
  echo "Creating disk ${ETCD_DISKS[${i}]}..."
  ${GCLOUD} compute disks create -q ${ETCD_DISKS[${i}]} \
      --zone=${ETCD_ZONES[${i}]} \
      --size=${ETCD_DISK_SIZE} &
done
wait

for i in `seq 0 $((${ETCD_NUM_REPLICAS} - 1))`; do
  echo "Waiting for disk ${ETCD_DISKS[${i}]}..."
  WaitForStatus disks ${ETCD_DISKS[${i}]} ${ETCD_ZONES[${i}]} READY &
done
wait

MANIFEST=/tmp/etcd_container.yaml
echo -n "Getting Discovery URL"
while [ "${DISCOVERY}" == "" ]; do
  DISCOVERY=$(curl -s https://discovery.etcd.io/new?size=${ETCD_NUM_REPLICAS})
  echo .
  sleep 1
done

echo
echo "Using Discovery URL: ${DISCOVERY}"
echo


Header "Creating etcd instances..."
for i in `seq 0 $((${ETCD_NUM_REPLICAS} - 1))`; do
  echo "Creating instance ${ETCD_MACHINES[$i]}"

  sed --e "s^@@PROJECT@@^${PROJECT}^
           s^@@DISCOVERY@@^${DISCOVERY}^
           s^@@ETCD_NAME@@^${ETCD_MACHINES[$i]}^
           s^@@CONTAINER_HOST@@^${ETCD_MACHINES[$i]}^" \
          < ${DIR}/etcd_container.yaml  > ${MANIFEST}.${i}

  ${GCLOUD} compute instances create -q ${ETCD_MACHINES[${i}]} \
      --zone ${ETCD_ZONES[${i}]} \
      --machine-type ${ETCD_MACHINE_TYPE} \
      --image container-vm \
      --disk name=${ETCD_DISKS[${i}]},mode=rw,boot=no,auto-delete=yes \
      --tags etcd-node \
      --metadata-from-file startup-script=${DIR}/node_init.sh,google-container-manifest=${MANIFEST}.${i} &
done
wait

for i in `seq 0 $((${ETCD_NUM_REPLICAS} - 1))`; do
  echo "Waiting for instance ${ETCD_MACHINES[${i}]}..."
  WaitForStatus instances ${ETCD_MACHINES[${i}]} ${ETCD_ZONES[${i}]} RUNNING &
done
wait


