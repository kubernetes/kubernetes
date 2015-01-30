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

# The business logic for whether a given object should be created
# was already enforced by salt, and /etc/kubernetes/addons is the
# managed result is of that. Start everything below that directory.
echo "== Kubernetes addon manager started at $(date -Is) =="
KUBECTL=/usr/local/bin/kubectl
for obj in $(find /etc/kubernetes/addons -name \*.yaml); do
  ${KUBECTL} --server="127.0.0.1:8080" create -f ${obj} &
  echo "++ addon ${obj} started in pid $! ++"
done
noerrors="true"
for pid in $(jobs -p); do
  wait ${pid} || noerrors="false"
  echo "++ pid ${pid} complete ++"
done
if [ ${noerrors} == "true" ]; then
  echo "== Kubernetes addon manager completed successfully at $(date -Is) =="
else
  echo "== Kubernetes addon manager completed with errors at $(date -Is) =="
fi

# We stay around so that status checks by salt make it look like
# the service is good. (We could do this is other ways, but this
# is simple.)
sleep infinity
