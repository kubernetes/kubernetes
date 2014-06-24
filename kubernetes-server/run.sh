#!/bin/sh
ETCD_SERVERS=$1
MACHINES=$2

if [ -z "$MACHINES " ]
then
  echo "$0 etcd-servers-http-urls machines-hostnames (both comma separated)"
  exit 1
fi

apiserver -address 0.0.0.0 -etcd_servers=$ETCD_SERVERS -machines=$MACHINES &
controller-manager -master localhost:8080 -etcd_servers=$ETCD_SERVERS
