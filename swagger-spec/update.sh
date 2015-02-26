#!/bin/bash

# Copyright 2015 Google Inc. All rights reserved.
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

# Script to update the swagger spec here by copying the spec from:
# https://rawgit.com/GoogleCloudPlatform/kubernetes/master/api/swagger-spec/
# We convert the json files from there to index.html files here to act as a
# server serving the spec for swagger-ui.

set -o errexit
set -o nounset
set -o pipefail

SOURCE_ROOT_PATH="https://rawgit.com/GoogleCloudPlatform/kubernetes/master/api/swagger-spec/"
DESTINATION_ROOT_PATH=$(dirname "${BASH_SOURCE}")/..
DESTINATION_PATH="${DESTINATION_ROOT_PATH}/swagger-spec"

echo "Fetching files from $SOURCE_ROOT_PATH and copying them to $DESTINATION_PATH"

curl $SOURCE_ROOT_PATH/resourceListing.json > $DESTINATION_PATH/index.html
curl $SOURCE_ROOT_PATH/api.json > $DESTINATION_PATH/api/index.html
curl $SOURCE_ROOT_PATH/v1beta1.json > $DESTINATION_PATH/api/v1beta1/index.html
curl $SOURCE_ROOT_PATH/v1beta2.json > $DESTINATION_PATH/api/v1beta2/index.html
curl $SOURCE_ROOT_PATH/v1beta3.json > $DESTINATION_PATH/api/v1beta3/index.html
curl $SOURCE_ROOT_PATH/version.json > $DESTINATION_PATH/version/index.html

echo "SUCCESS!!"
