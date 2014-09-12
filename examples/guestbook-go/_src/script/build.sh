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

# Usage: ./script/build.sh

set -o errexit
set -o nounset
set -o pipefail

if [[ "${DOCKER_BIN+set}" == "set" ]]; then
  echo "Using DOCKER_BIN=\"${DOCKER_BIN}\" from the environment."
elif DOCKER_BIN=$(which docker); then
  echo "Setting DOCKER_BIN=\"${DOCKER_BIN}\" from host machine."
else
  echo "Could not find a working docker binary and none passed in DOCKER_BIN." >&2
  exit 1
fi

docker build --rm --force-rm -t kubernetes/guestbook-build .
docker run --rm -v "${DOCKER_BIN}:/usr/local/bin/docker" \
                -v "/var/run/docker.sock:/var/run/docker.sock" \
                -ti --name guestbook-build kubernetes/guestbook-build
