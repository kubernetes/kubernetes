#!/usr/bin/env bash

# Copyright 2016 The Kubernetes Authors.
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

# A script that let's gci preemptible nodes gracefully terminate in the event of a VM shutdown.
preemptible=$(curl "http://metadata.google.internal/computeMetadata/v1/instance/scheduling/preemptible" -H "Metadata-Flavor: Google")
if [ "${preemptible}" == "TRUE" ]; then
    echo "Shutting down! Sleeping for a minute to let the node gracefully terminate"
    # https://cloud.google.com/compute/docs/instances/stopping-or-deleting-an-instance#delete_timeout
    sleep 30
fi
