# Copyright 2018 The Kubernetes Authors.
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

# The e2e aggregator test is designed to test ability to run sample-apiserver as an aggregated server.
# see e2e test named "Should be able to support the 1.17 Sample API Server using the current Aggregator"

ARG BASEIMAGE
FROM $BASEIMAGE
ADD sample-apiserver /sample-apiserver

ENTRYPOINT ["/sample-apiserver"]
