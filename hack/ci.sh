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

# Travis doesn't currently support go on OS X
if [ "$TRAVIS" == "true" ] && [ "$CI" == "true" ] && [ "$(uname)" == "Darwin" ]; then
	exit 0
fi

go get code.google.com/p/go.tools/cmd/cover
go get github.com/coreos/etcd
