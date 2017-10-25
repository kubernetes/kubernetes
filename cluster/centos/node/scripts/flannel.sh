#!/bin/bash

# Copyright 2014 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


ETCD_SERVERS=${1:-"https://8.8.8.18:2379"}
FLANNEL_NET=${2:-"172.16.0.0/16"}

CA_FILE="/srv/kubernetes/etcd/ca.pem"
CERT_FILE="/srv/kubernetes/etcd/client.pem"
KEY_FILE="/srv/kubernetes/etcd/client-key.pem"

cat <<EOF >/opt/kubernetes/cfg/flannel
FLANNEL_ETCD="-etcd-endpoints=${ETCD_SERVERS}"
FLANNEL_ETCD_KEY="-etcd-prefix=/coreos.com/network"
FLANNEL_ETCD_CAFILE="--etcd-cafile=${CA_FILE}"
FLANNEL_ETCD_CERTFILE="--etcd-certfile=${CERT_FILE}"
FLANNEL_ETCD_KEYFILE="--etcd-keyfile=${KEY_FILE}"
EOF

cat <<EOF >/usr/lib/systemd/system/flannel.service
[Unit]
Description=Flanneld overlay address etcd agent
After=network.target
Before=docker.service

[Service]
EnvironmentFile=-/opt/kubernetes/cfg/flannel
ExecStartPre=/opt/kubernetes/bin/remove-docker0.sh
ExecStart=/opt/kubernetes/bin/flanneld --ip-masq \${FLANNEL_ETCD} \${FLANNEL_ETCD_KEY} \${FLANNEL_ETCD_CAFILE} \${FLANNEL_ETCD_CERTFILE} \${FLANNEL_ETCD_KEYFILE}
ExecStartPost=/opt/kubernetes/bin/mk-docker-opts.sh -d /run/flannel/docker

Type=notify

[Install]
WantedBy=multi-user.target
RequiredBy=docker.service
EOF

# Store FLANNEL_NET to etcd.
attempt=0
while true; do
  /opt/kubernetes/bin/etcdctl --ca-file ${CA_FILE} --cert-file ${CERT_FILE} --key-file ${KEY_FILE} \
    --no-sync -C ${ETCD_SERVERS} \
    get /coreos.com/network/config >/dev/null 2>&1
  if [[ "$?" == 0 ]]; then
    break
  else
    if (( attempt > 600 )); then
      echo "timeout for waiting network config" > ~/kube/err.log
      exit 2
    fi

    /opt/kubernetes/bin/etcdctl --ca-file ${CA_FILE} --cert-file ${CERT_FILE} --key-file ${KEY_FILE} \
      --no-sync -C ${ETCD_SERVERS} \
      mk /coreos.com/network/config "{\"Network\":\"${FLANNEL_NET}\"}" >/dev/null 2>&1
    attempt=$((attempt+1))
    sleep 3
  fi
done
wait

systemctl daemon-reload
