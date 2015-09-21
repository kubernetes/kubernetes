#!/bin/bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
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

# This does a checked upgrade of the MASTER using the locally built release, then runs e2e.
echo "Running the checked master upgrade."
go run "$(dirname $0)/e2e.go" -build -up -v -test -test_args='--ginkgo.focus=Skipped.*Cluster\supgrade.*gce-upgrade' -check_version_skew=false
if [ $? -eq 0 ]; then
    echo "Master upgrade complete. Running e2e on the upgraded cluster."
    go run "$(dirname $0)/e2e.go" -v -version="" -test -check_version_skew=false
else
    echo "Master upgrade failed."
fi

exit $?
