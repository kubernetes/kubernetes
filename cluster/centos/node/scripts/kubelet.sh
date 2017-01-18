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

MASTER_ADDRESS=${1:-"8.8.8.18"}
NODE_ADDRESS=${2:-"8.8.8.20"}
DNS_SERVER_IP=${3:-"192.168.3.100"}
DNS_DOMAIN=${4:-"cluster.local"}
KUBECONFIG_DIR=${KUBECONFIG_DIR:-/opt/kubernetes/cfg}

# Generate a kubeconfig file
cat <<EOF > "${KUBECONFIG_DIR}/kubelet.kubeconfig"
apiVersion: v1
kind: Config
clusters:
  - cluster:
      server: http://${MASTER_ADDRESS}:8080/
    name: local
contexts:
  - context:
      cluster: local
    name: local
current-context: local
EOF

cat <<EOF >/opt/kubernetes/cfg/kubelet
# --logtostderr=true: log to standard error instead of files
KUBE_LOGTOSTDERR="--logtostderr=true"

#  --v=0: log level for V logs
KUBE_LOG_LEVEL="--v=4"

# --address=0.0.0.0: The IP address for the Kubelet to serve on (set to 0.0.0.0 for all interfaces)
NODE_ADDRESS="--address=${NODE_ADDRESS}"

# --port=10250: The port for the Kubelet to serve on. Note that "kubectl logs" will not work if you set this flag.
NODE_PORT="--port=10250"

# --hostname-override="": If non-empty, will use this string as identification instead of the actual hostname.
NODE_HOSTNAME="--hostname-override=${NODE_ADDRESS}"

# Path to a kubeconfig file, specifying how to connect to the API server.
KUBELET_KUBECONFIG="--kubeconfig=${KUBECONFIG_DIR}/kubelet.kubeconfig"

# --allow-privileged=false: If true, allow containers to request privileged mode. [default=false]
KUBE_ALLOW_PRIV="--allow-privileged=false"

# DNS info
KUBELET__DNS_IP="--cluster-dns=${DNS_SERVER_IP}"
KUBELET_DNS_DOMAIN="--cluster-domain=${DNS_DOMAIN}"

# Add your own!
KUBELET_ARGS=""
EOF

KUBELET_OPTS="      \${KUBE_LOGTOSTDERR}     \\
                    \${KUBE_LOG_LEVEL}       \\
                    \${NODE_ADDRESS}         \\
                    \${NODE_PORT}            \\
                    \${NODE_HOSTNAME}        \\
                    \${KUBELET_KUBECONFIG}   \\
                    \${KUBE_ALLOW_PRIV}      \\
                    \${KUBELET__DNS_IP}      \\
                    \${KUBELET_DNS_DOMAIN}   \\
                    \$KUBELET_ARGS"

cat <<EOF >/usr/lib/systemd/system/kubelet.service
[Unit]
Description=Kubernetes Kubelet
After=docker.service
Requires=docker.service

[Service]
EnvironmentFile=-/opt/kubernetes/cfg/kubelet
ExecStart=/opt/kubernetes/bin/kubelet ${KUBELET_OPTS}
Restart=on-failure
KillMode=process

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable kubelet
systemctl restart kubelet
