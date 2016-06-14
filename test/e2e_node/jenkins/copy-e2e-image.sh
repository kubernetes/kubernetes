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

# Usage: copy-e2e-image.sh <image-name> <from-project-name> <to-project-name>

# See *.properties for list of images to copy,
# typically from kubernetes-node-e2e-images

set -e
set -x

echo "Copying image $1 from project $2 to project $3..."
gcloud compute --project $3 disks create $1 --image=https://www.googleapis.com/compute/v1/projects/$2/global/images/$1
gcloud compute --project $3 images create $1 \
  --source-disk=$1 \
  --description="Cloned from projects/$2/global/images/$1 by $USER on $(date)"
gcloud -q compute --project $3 disks delete $1
