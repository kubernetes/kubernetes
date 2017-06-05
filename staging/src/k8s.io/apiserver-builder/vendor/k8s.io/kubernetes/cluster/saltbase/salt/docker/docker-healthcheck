#!/bin/bash

# Copyright 2015 The Kubernetes Authors.
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

# This script is intended to be run periodically, to check the health
# of docker.  If it detects a failure, it will restart docker using systemctl.

if timeout 10 docker version > /dev/null; then
  exit 0
fi

echo "docker failed"
echo "Giving docker 30 seconds grace before restarting"
sleep 30

if timeout 10 docker version > /dev/null; then
  echo "docker recovered"
  exit 0
fi

echo "docker still down; triggering docker restart"
systemctl restart docker

echo "Waiting 60 seconds to give docker time to start"
sleep 60

if timeout 10 docker version > /dev/null; then
  echo "docker recovered"
  exit 0
fi

echo "docker still failing"
