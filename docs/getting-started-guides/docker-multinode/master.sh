#!/bin/bash

# Copyright 2016 The Kubernetes Authors All rights reserved.
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

# Source common.sh
source $(dirname "${BASH_SOURCE}")/common.sh

# Let MASTER_IP default to the current IP when starting a master
if [[ -z ${MASTER_IP} ]]; then
  MASTER_IP=$(hostname -I | awk '{print $1}')
fi

kube::multinode::check_params

kube::multinode::detect_lsb

kube::multinode::turndown

kube::multinode::bootstrap_daemon

kube::multinode::start_etcd

kube::multinode::start_flannel

kube::multinode::restart_docker

kube::multinode::start_k8s_master

kube::log::status "Done. It will take some minutes before apiserver is up though"