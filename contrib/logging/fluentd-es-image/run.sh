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

# WARNING! HORRIBLE HACK! We expect /outerhost to be mapped to
# the enclosing /etc/host file so we can determine the name of
# the host machine (super fragile). This is a temporary hack until
# service IPs are done.
OUTER_HOST=`tail -n 1 /outerhost | awk '{print $3}'`

# WARNING! HORRIBLE HACK! We expect /outerhost to be mapped to
# the enclosing /etc/host file so we can determine the name of
# the host machine (super fragile). This is a temporary hack until
# service IPs are done.
# Adjust the name of the host machine for %ES_HOST%. HACK!
sed -i -e "s/\%ES_HOST\%/${OUTER_HOST}/" /etc/td-agent/td-agent.conf
/usr/sbin/td-agent "$@"
