#!/bin/bash

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

# Tear down a Kubernetes cluster.

# exit on any error
set -e

source $(dirname $0)/util.sh

# Detect the project into $PROJECT
detect-project

echo "Bringing down cluster"
gcutil deletefirewall  \
  --project ${PROJECT} \
  --norespect_terminal_width \
  --force \
  ${MASTER_NAME}-https &

gcutil deleteinstance \
  --project ${PROJECT} \
  --norespect_terminal_width \
  --force \
  --delete_boot_pd \
  --zone ${ZONE} \
  ${MASTER_NAME} &

gcutil deleteinstance \
  --project ${PROJECT} \
  --norespect_terminal_width \
  --force \
  --delete_boot_pd \
  --zone ${ZONE} \
  ${MINION_NAMES[*]} &

gcutil deleteroute  \
  --project ${PROJECT} \
  --force \
  ${MINION_NAMES[*]} &

wait
