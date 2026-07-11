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

METADATA_ENDPOINT="http://metadata.google.internal/computeMetadata/v1/instance/attributes/kube-master-internal-ip"
METADATA_HEADER="Metadata-Flavor: Google"
ip=$(curl -s --fail ${METADATA_ENDPOINT} -H "${METADATA_HEADER}")
if [ -n "$ip" ];
then
    # Check if route is already set if not set it
    if ! sudo ip route show table local | grep -q "$(echo "$ip" | cut -d'/' -f 1)";
    then
            sudo ip route add to local "${ip}/32" dev "$(ip route | grep default | awk '{print $5}')"
    fi
fi
