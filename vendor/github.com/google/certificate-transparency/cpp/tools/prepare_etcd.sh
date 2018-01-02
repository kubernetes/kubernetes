#!/bin/bash
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)

if [ $# -ne 3 ]; then
  echo "Usage $0 <etcd host> <etcd port> <log key pem>"
  exit 1;
fi

ETCD_HOST=$1
ETCD_PORT=$2
LOG_KEY=$3

ETCD=http://${ETCD_HOST}:${ETCD_PORT}
curl -L -X PUT ${ETCD}/v2/keys/root -d dir=true
curl -L -X PUT ${ETCD}/v2/keys/root/entries -d dir=true
curl -L -X PUT ${ETCD}/v2/keys/root/nodes -d dir=true
curl -L -X PUT ${ETCD}/v2/keys/root/serving_sth
curl -L -X PUT ${ETCD}/v2/keys/root/cluster_config
curl -L -X PUT ${ETCD}/v2/keys/root/sequence_mapping
${DIR}/ct-clustertool initlog \
    --key=${LOG_KEY} \
    --etcd_servers="${ETCD_HOST}:${ETCD_PORT}" \
    --logtostderr
