# Copyright 2016 The Kubernetes Authors.
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

# TODO - At some point, we need to rename all skydns-*.yaml.* files to kubedns-*.yaml.*

# Warning: This is a file generated from the base underscore template file: skydns-scaler-rc.yaml.base

kind: ReplicationController
apiVersion: v1
metadata:
  name: kube-dns-autoscaler-v19
  namespace: kube-system
  labels:
    k8s-app: kube-dns-autoscaler
    version: v19
    kubernetes.io/cluster-service: "true"
spec:
  replicas: 1
  selector:
    k8s-app: kube-dns-autoscaler
    version: v19
  template:
    metadata:
      labels:
        k8s-app: kube-dns-autoscaler
        version: v19
        kubernetes.io/cluster-service: "true"
    spec:
      containers:
      - name: pod-autoscaler-nanny
        image: gcr.io/google_containers/dns-rc-autoscaler:0.6
        resources:
          limits:
            cpu: 10m
            memory: 200Mi
          requests:
            cpu: 10m
            memory: 40Mi
        command:
          - /dns_pod_nanny
          - --configmap
          - pod-autoscaler-nanny-params
          - --namespace
          - default
          - --rc
          - kube-dns-v19
