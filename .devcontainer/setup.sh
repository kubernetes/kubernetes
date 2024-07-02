#!/usr/bin/env bash

# Copyright 2024 The Kubernetes Authors.
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

set -eux

# Copies over welcome message
mkdir -p /usr/local/etc/vscode-dev-containers
cp .devcontainer/welcome-message.txt /usr/local/etc/vscode-dev-containers/first-run-notice.txt

# Ensure that the upstream remote is set, and configure it if it's not
if ! git remote | grep -q "^upstream$"; then
    git remote add upstream https://github.com/kubernetes/kubernetes.git
else
    if [ "$(git remote get-url upstream)" != "https://github.com/kubernetes/kubernetes.git" ]; then
      git remote set-url upstream https://github.com/kubernetes/kubernetes.git
    fi
fi

# Never push to upstream master
git remote set-url --push upstream no_push

# To ensure success in hack/print-workspace-status.sh,
# execute git fetch upstream beforehand
# The information retrieved in hack/print-workspace-status.sh is necessary
# to execute the 'kind build node-image' command successfully through kubetest2
# https://github.com/kubernetes-sigs/kind/blob/v0.22.0/pkg/build/nodeimage/internal/kube/source.go#L71-L104
git fetch upstream

# Install etcd
./hack/install-etcd.sh

# Install gcloud command
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
apt-get update
apt-get install -yq google-cloud-cli
rm -f /etc/apt/sources.list.d/google-cloud-sdk.list

# Install PyYAML
pip3 install pyyaml

# Install kind & kubetest2 for e2e testing
go install sigs.k8s.io/kind@latest
go install sigs.k8s.io/kubetest2/...@latest
