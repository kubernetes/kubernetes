#!/bin/bash

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

# Vars assumed:
#   NUM_NODES
function get-master-size {
  local suggested_master_size=1
  if [[ "${NUM_NODES}" -gt "5" ]]; then
    suggested_master_size=2
  fi
  if [[ "${NUM_NODES}" -gt "10" ]]; then
    suggested_master_size=4
  fi
  if [[ "${NUM_NODES}" -gt "100" ]]; then
    suggested_master_size=8
  fi
  if [[ "${NUM_NODES}" -gt "250" ]]; then
    suggested_master_size=16
  fi
  if [[ "${NUM_NODES}" -gt "500" ]]; then
    suggested_master_size=32
  fi
  echo "${suggested_master_size}"
}

if [[ "${FEDERATION:-}" == true ]]; then
    NODE_SCOPES="${NODE_SCOPES:-compute-rw,monitoring,logging-write,storage-ro,https://www.googleapis.com/auth/ndev.clouddns.readwrite}"
else
    NODE_SCOPES="${NODE_SCOPES:-compute-rw,monitoring,logging-write,storage-ro}"
fi
