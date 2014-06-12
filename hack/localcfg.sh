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

# This file is exactly like cloudcfg.sh, but it talks to a local master
# (which you're assumed to be running with localkube.sh).

CLOUDCFG=$(dirname $0)/../output/go/cloudcfg
if [ ! -x $CLOUDCFG ]; then
  echo "Could not find cloudcfg binary. Run hack/build-go.sh to build it."
  exit 1
fi

# 8080 is the default port for the master
$CLOUDCFG -h http://localhost:8080 $@
