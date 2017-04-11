#!/usr/bin/env python

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

import json
import sys
# This script helps parse out the private IP addresses from the
# `juju run` command's JSON object, see cluster/juju/util.sh

if len(sys.argv) > 1:
    # It takes the JSON output as the first argument.
    nodes = json.loads(sys.argv[1])
    # There can be multiple nodes to print the Stdout.
    for num in nodes:
        print num['Stdout'].rstrip()
else:
    exit(1)
