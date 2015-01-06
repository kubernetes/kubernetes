#!/bin/bash

# Copyright 2014 Google Inc. All rights reserved.
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

# see https://github.com/docker/docker/issues/8395
HOST_IP=$(ip route show 0.0.0.0/0 | grep -Eo 'via \S+' | awk '{ print $2 }')

KUBELET_IP=$(hostname -i)

cat <<EOF > pod.yaml
version: v1beta1
id: cluster-pod
containers:
  - name: etcd
    image: coreos/etcd
    command: ["/go/bin/app", "-vv", "-listen-client-urls=http://0.0.0.0:4001"]
    ports:
      - name: etcd-port
        hostPort: 4001
        containerPort: 4001
        protocol: TCP
  - name: kube-apiserver
    image: kubernetes
    imagePullPolicy: never
    ports:
      - name: kube-apiserver-port
        hostPort: 8080
        containerPort: 8080
        protocol: TCP
    command: ["/kubernetes/kube-apiserver", "-v=5", "-address=0.0.0.0", "-etcd_servers=http://127.0.0.1:4001", "-portal_net=10.0.0.0/24"]
  - name: controller-manager
    image: kubernetes
    imagePullPolicy: never
    command: ["/kubernetes/kube-controller-manager", "-v=5", "-master=127.0.0.1:8080", "-machines=${KUBELET_IP}"]
  - name: proxy
    image: kubernetes
    imagePullPolicy: never
    command: ["/kubernetes/kube-proxy", "-v=5", "-etcd_servers=http://127.0.0.1:4001"]
  - name: scheduler
    image: kubernetes
    imagePullPolicy: never
    command: ["/kubernetes/kube-scheduler", "-v=5", "-master=127.0.0.1:8080"]
EOF
./kubelet -v=5 -address=0.0.0.0 -hostname_override=${KUBELET_IP} -etcd_servers=http://${HOST_IP}:4001 -config pod.yaml
