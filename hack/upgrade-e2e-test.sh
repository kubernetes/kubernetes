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

# Provide reasonable default for running the end-to-end tests against a recent
# stable release and then again after upgrading it to a version built from head.

go run "$(dirname $0)/e2e.go" -v -build -up -version="v0.14.0" -test -check_version_skew=false
if [ $? -eq 0 ]; then
	echo "Tests on initial version succeeded. Proceeding with push and second set of tests."
	go run "$(dirname $0)/e2e.go" -v -push -version="" -test -check_version_skew=false
else
	echo "Tests on initial version failed. Skipping tests on second version."
fi

exit $?
