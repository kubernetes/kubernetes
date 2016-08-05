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

# loadedImageFlags is a bit-flag to track which docker images loaded successfully.
let loadedImageFlags=0

while true; do
  restart_docker=false

  if which docker 1>/dev/null 2>&1; then

    timeout 120 docker load -i /srv/salt/kube-bins/kube-proxy.tar 1>/dev/null 2>&1
    rc=$?
    if [[ "${rc}" == 0 ]]; then
      let loadedImageFlags="${loadedImageFlags}|1"
    elif [[ "${rc}" == 124 ]]; then
      restart_docker=true
    fi
  fi

  # required docker images got installed. exit while loop.
  if [[ "${loadedImageFlags}" == 1 ]]; then break; fi

  # Sometimes docker load hang, restart docker daemon resolve the issue
  if [[ "${restart_docker}" ]]; then service docker restart; fi

  # sleep for 15 seconds before attempting to load docker images again
  sleep 15

done

# Now exit. After kube-push, salt will notice that the service is down and it
# will start it and new docker images will be loaded.
