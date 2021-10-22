#!/bin/bash -e

# Copyright 2017 VMware, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

export GOVC_INSECURE=true

govc license.add "$VCSA_LICENSE" "$ESX_LICENSE" >/dev/null

govc license.assign "$VCSA_LICENSE" >/dev/null

govc find / -type h | xargs -I% -n1 govc license.assign -host % "$ESX_LICENSE" >/dev/null

echo "Assigned licenses..."
govc license.assigned.ls

echo ""
echo "License usage..."
govc license.ls
