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

# This script will build and push the images necessary for the demo.

# If a user is provided, then use it. If not, environment var must be set.
if [ $# -eq 1 ] ; then
  DOCKER_HUB_USER=$1
elif [ -z "$DOCKER_HUB_USER" ] ; then
  echo "Usage: $0 <docker hub user name>"
  exit 1
fi

set -x

sudo docker build -t update-demo-base images/base
sudo docker build -t $DOCKER_HUB_USER/update-demo:kitten images/kitten
sudo docker build -t $DOCKER_HUB_USER/update-demo:nautilus images/nautilus

sudo docker push $DOCKER_HUB_USER/update-demo
