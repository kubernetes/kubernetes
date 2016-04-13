#!/usr/bin/env bash

# Copyright 2016 Google Inc. All rights reserved.
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

if [[ -n "${JENKINS_HOME}" ]]
then 
    exec ./build/jenkins_e2e.sh
fi
echo ">> starting cAdvisor locally"
sudo ./cadvisor &
echo ">> running integration tests against local cAdvisor"
godep go test github.com/google/cadvisor/integration/tests/... --vmodule=*=2
if [ $? -ne 0 ]
then
    echo "Integration tests failed"
fi
echo ">> stopping cAdvisor"
sudo pkill -9 cadvisor

