#!/usr/bin/env bash

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

# Run the CockroachDB StatefulSet example on a minikube instance.
#
# For a fresh start, run the following first:
#   minikube delete
#   minikube start
#
# To upgrade minikube & kubectl on OSX, the following should suffice:
#   brew reinstall kubernetes-cli --devel
#   url -Lo minikube \
#     https://storage.googleapis.com/minikube/releases/v0.4.0/minikube-darwin-amd64 && \
#   chmod +x minikube && sudo mv minikube /usr/local/bin/

set -exuo pipefail

# Clean up anything from a prior run:
kubectl delete statefulsets,persistentvolumes,persistentvolumeclaims,services,poddisruptionbudget -l app=cockroachdb

# Make persistent volumes and (correctly named) claims. We must create the
# claims here manually even though that sounds counter-intuitive. For details
# see https://github.com/kubernetes/contrib/pull/1295#issuecomment-230180894.
# Note that we make an extra volume here so you can manually test scale-up.
for i in $(seq 0 3); do
  cat <<EOF | kubectl create -f -
kind: PersistentVolume
apiVersion: v1
metadata:
  name: pv${i}
  labels:
    type: local
    app: cockroachdb
spec:
  capacity:
    storage: 1Gi
  accessModes:
    - ReadWriteOnce
  hostPath:
    path: "/tmp/${i}"
EOF

  cat <<EOF | kubectl create -f -
kind: PersistentVolumeClaim
apiVersion: v1
metadata:
  name: datadir-cockroachdb-${i}
  labels:
    app: cockroachdb
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
EOF
done;

kubectl create -f cockroachdb-statefulset.yaml
