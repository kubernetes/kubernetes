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

# Make all of the Kubernetes binaries for cross compile targets
#
# This makes the docker build image, builds the cross binaries and copies them
# out of the docker container.

set -e

source $(dirname $0)/common.sh

verify-prereqs
build-image
run-build-command build/build-image/make-cross.sh
