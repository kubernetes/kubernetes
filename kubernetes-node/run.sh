#!/bin/sh
set -e
if [ -z "$3" ]
then
  echo "$0 peer-addr addr discovery-url"
  exit 1
fi

ID="ip_`echo $1 | cut -d: -f1|tr '.' '_'`"
echo "ID=$ID"
kubelet -config /etc/kubelet.conf -docker unix:///docker.sock \
        -address="0.0.0.0" -hostname_override=$ID -etcd_servers="http://localhost:4001" &

/etcd/etcd -peer-addr $1 -addr $2 -discovery $3
