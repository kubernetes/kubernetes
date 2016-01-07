#!/bin/bash

# Copyright 2015 The Kubernetes Authors All rights reserved.
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

PROJECT_ID=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/project/project-id)

if [[ -n "${PROJECT_ID}" ]]; then
  sed -i "s/NOT_RUNNING_INSIDE_GCE/${PROJECT_ID}/" /opt/spark/conf/core-site.xml
fi

# We don't want any of the incoming service variables, we'd rather use
# DNS. But this one interferes directly with Spark.
unset SPARK_MASTER_PORT

# spark.{executor,driver}.extraLibraryPath don't actually seem to
# work, this seems to be the only reliable way to get the native libs
# picked up.
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/hadoop/lib/native
