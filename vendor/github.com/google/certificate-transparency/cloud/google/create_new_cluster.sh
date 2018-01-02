#!/bin/bash
set -e
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
GCLOUD="gcloud"
if [ "$1" == "" ]; then
  echo "Usage: $0 <config-file>"
  exit 1
fi
CONFIG_FILE="$1"

. ${DIR}/config.sh ${CONFIG_FILE}

if [ ! -x ${DIR}/../../cpp/tools/ct-clustertool ]; then
  echo "Please ensure that cpp/tools/ct-clustertool is built."
  exit 1
fi

function WaitForEtcd() {
  echo "Waiting for etcd @ ${ETCD_MACHINES[1]}"
  while true; do
    gcloud compute ssh ${ETCD_MACHINES[1]} \
        --zone ${ETCD_ZONES[1]} \
        --command "\
     until curl -s -L -m 10 localhost:4001/v2/keys/ > /dev/null; do \
       echo -n .; \
       sleep 1; \
     done" && break;
    sleep 1
    echo "Retrying..."
  done
}

function PopulateEtcdForLog() {
  export PUT="curl -s -L -X PUT --retry 10"
  export ETCD="${ETCD_MACHINES[1]}:4001"
  gcloud compute ssh ${ETCD_MACHINES[1]} \
      --zone ${ETCD_ZONES[1]} \
      --command "\
    ${PUT} ${ETCD}/v2/keys/root/serving_sth && \
    ${PUT} ${ETCD}/v2/keys/root/cluster_config && \
    ${PUT} ${ETCD}/v2/keys/root/sequence_mapping && \
    ${PUT} ${ETCD}/v2/keys/root/entries/ -d dir=true && \
    ${PUT} ${ETCD}/v2/keys/root/nodes/ -d dir=true"

  gcloud compute ssh ${ETCD_MACHINES[1]} \
      --zone ${ETCD_ZONES[1]} \
      --command "\
    sudo docker run gcr.io/${PROJECT}/super_duper:test \
      /usr/local/bin/ct-clustertool initlog \
      --key=/usr/local/etc/server-key.pem \
      --etcd_servers=${ETCD_MACHINES[1]}:4001 \
      --logtostderr"
}

function PopulateEtcdForMirror() {
  export PUT="curl -s -L -X PUT --retry 10"
  export ETCD="${ETCD_MACHINES[1]}:4001"
  gcloud compute ssh ${ETCD_MACHINES[1]} \
      --zone ${ETCD_ZONES[1]} \
      --command "\
    ${PUT} ${ETCD}/v2/keys/root/serving_sth && \
    ${PUT} ${ETCD}/v2/keys/root/cluster_config && \
    ${PUT} ${ETCD}/v2/keys/root/nodes/ -d dir=true"
}


echo "============================================================="
echo "Creating new GCE-based ${INSTANCE_TYPE} cluster."
echo "============================================================="

# Set gcloud defaults:
${GCLOUD} config set project ${PROJECT}

echo "============================================================="
echo "Creating etcd instances..."
${DIR}/start_etcd.sh ${CONFIG_FILE}

WaitForEtcd

echo "============================================================="
echo "Populating etcd with default entries..."
case "${INSTANCE_TYPE}" in
  "log")
    PopulateEtcdForLog
    ;;
  "mirror")
    PopulateEtcdForMirror
    ;;
  *)
    echo "Unknown INSTANCE_TYPE: ${INSTANCE_TYPE}"
    exit 1
esac

echo "============================================================="
echo "Creating superduper ${INSTANCE_TYPE} instances..."
case "${INSTANCE_TYPE}" in
  "log")
    ${DIR}/start_log.sh ${CONFIG_FILE}
    ;;
  "mirror")
    ${DIR}/start_mirror.sh ${CONFIG_FILE}
    ;;
  *)
    echo "Unknown INSTANCE_TYPE: ${INSTANCE_TYPE}"
    exit 1
esac

if [ "${MONITORING}" == "prometheus" ]; then
  echo "============================================================="
  echo "Starting prometheus..."
  ${DIR}/start_prometheus.sh ${CONFIG_FILE}
  ${DIR}/update_prometheus_config.sh ${CONFIG_FILE}
fi

${DIR}/configure_service.sh ${CONFIG_FILE}

echo "Job done!"
