#! /bin/bash

# Copyright 2014 Google Inc. All rights reserved.
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

# If the user doesn't specify a minion, assume we are running in a single node
# configuration and that we have a local minion.
KUBE_MINIONS="${KUBE_MINIONS:-$(hostname -f)}"

./apiserver -address=0.0.0.0 -etcd_servers="${ETCD_SERVERS}" --machines="${KUBE_MINIONS}"
