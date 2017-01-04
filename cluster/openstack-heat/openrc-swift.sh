#!/usr/bin/env bash

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

## Enviroment variables for the OpenStack Swift command-line client. This is required for CityCloud
## provider where Swift has different credentials. When Swift is part of your OpenStack do not
## modify these settings.

export OS_IDENTITY_API_VERSION=${OS_IDENTITY_API_VERSION:-2.0}
export OS_USERNAME=${OS_USERNAME:-admin}
export OS_PASSWORD=${OS_PASSWORD:-secretsecret}
export OS_AUTH_URL=${OS_AUTH_URL:-http://192.168.123.100:5000/v2.0}
export OS_TENANT_NAME=${OS_TENANT_NAME:-admin}
export OS_REGION_NAME=${OS_REGION_NAME:-RegionOne}
