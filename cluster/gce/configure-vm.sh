#!/bin/bash

# Copyright 2015 The Kubernetes Authors All rights reserved.
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

set -o errexit
set -o nounset
set -o pipefail

# If we have any arguments at all, this is a push and not just setup.
is_push=$@

function download-and-source-common-config() {
  mkdir -p /etc/kubernetes/
  local dst=/etc/kubernetes/configure-common.sh
  curl -H "X-Google-Metadata-Request: True" \
       -o ${dst} \
       http://metadata.google.internal/computeMetadata/v1/instance/attributes/configure-common
  source ${dst}
}

####################################################################################

download-and-source-common-config

if [[ -z "${is_push}" ]]; then
  echo "== kube-up node config starting =="
  set-broken-motd
  ensure-basic-networking
  ensure-install-dir
  set-kube-env
  [[ "${KUBERNETES_MASTER}" == "true" ]] && mount-master-pd
  create-salt-pillar
  if [[ "${KUBERNETES_MASTER}" == "true" ]]; then
    create-salt-master-auth
    create-salt-master-kubelet-auth
  else
    create-salt-kubelet-auth
    create-salt-kubeproxy-auth
  fi
  download-release
  configure-salt
  remove-docker-artifacts
  run-salt
  set-good-motd
  echo "== kube-up node config done =="
else
  echo "== kube-push node config starting =="
  ensure-basic-networking
  ensure-install-dir
  set-kube-env
  create-salt-pillar
  download-release
  run-salt
  echo "== kube-push node config done =="
fi
