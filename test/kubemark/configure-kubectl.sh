#!/bin/bash

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

curl https://sdk.cloud.google.com 2> /dev/null | bash
sudo gcloud components update kubectl -q
sudo ln -s /usr/local/share/google/google-cloud-sdk/bin/kubectl /bin/
kubectl config set-cluster hollow-cluster --server=http://localhost:8080 --insecure-skip-tls-verify=true
kubectl config set-credentials $(whoami)
kubectl config set-context hollow-context --cluster=hollow-cluster --user=$(whoami)
