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

# Create generic token following GCE standard
create_token() {
  echo $(cat /dev/urandom | base64 | tr -d "=+/" | dd bs=32 count=1 2> /dev/null)
}

get_tokens_from_csv() {
  KUBE_BEARER_TOKEN=$(awk -F, '/admin/ {print $1}' ${KUBE_TEMP}/${1}_tokens.csv)
  KUBELET_TOKEN=$(awk -F, '/kubelet/ {print $1}' ${KUBE_TEMP}/${1}_tokens.csv)
  KUBE_PROXY_TOKEN=$(awk -F, '/kube_proxy/ {print $1}' ${KUBE_TEMP}/${1}_tokens.csv)
}

generate_admin_token() {
  echo "$(create_token),admin,admin" >> ${KUBE_TEMP}/known_tokens.csv
}

# Creates a csv file each time called (i.e one per kubelet).
generate_kubelet_tokens() {
  echo "$(create_token),kubelet,kubelet" > ${KUBE_TEMP}/${1}_tokens.csv
  echo "$(create_token),kube_proxy,kube_proxy" >> ${KUBE_TEMP}/${1}_tokens.csv
}
