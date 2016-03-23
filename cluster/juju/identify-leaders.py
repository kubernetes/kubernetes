#!/usr/bin/env python

# Copyright 2016 The Kubernetes Authors All rights reserved.
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

from subprocess import check_output
import yaml

out = check_output(['juju', 'status', 'kubernetes', '--format=yaml'])
try:
    parsed_output = yaml.safe_load(out)
    model = parsed_output['services']['kubernetes']['units']
    for unit in model:
        if 'workload-status' in model[unit].keys():
            if 'leader' in model[unit]['workload-status']['message']:
                print(unit)
except:
    pass
