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

# Tears down an existing cluster.  Warning destroys _all_ docker containers on the machine

set -o errexit
set -o nounset
set -o pipefail

echo "Warning, this will delete all Docker containers on this machine."
echo "Proceed? [Y/n]"

read resp
if [[ $resp == "n" || $resp == "N" ]]; then
  exit 0
fi

docker ps -aq | xargs docker rm -f
