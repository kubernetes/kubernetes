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

# This is an example script that stops vtctld.

set -e

script_root=`dirname "${BASH_SOURCE}"`
source $script_root/env.sh

echo "Stopping vtctld replicationcontroller..."
$KUBECTL stop replicationcontroller vtctld

echo "Deleting vtctld service..."
$KUBECTL delete service vtctld
