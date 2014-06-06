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

if [ "$(which etcd)" == "" ]; then
	echo "etcd must be in your PATH"
	exit 1
fi

# Stop right away if the build fails
set -e

./src/scripts/build-go.sh

etcd -name test -data-dir /tmp/foo > /tmp/etcd.log &

sleep 5

./target/integration

killall etcd
rm -rf /tmp/foo
