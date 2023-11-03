#!/usr/bin/env bash
# Copyright 2015 The Kubernetes Authors.
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

# only supports Linux

set -eux

# Clean up
rm -rf /var/lib/apt/lists/*

echo "Installing kubetest2..."
go install sigs.k8s.io/kubetest2/...@latest

apt-get update
apt-get -y install --no-install-recommends curl apt-transport-https ca-certificates gnupg
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.gpg
apt-get update && apt-get install google-cloud-cli

chown -R "${_REMOTE_USER}:golang" "${GOPATH}"
chmod -R g+r+w "${GOPATH}"
find "${GOPATH}" -type d -print0 | xargs -n 1 -0 chmod g+s