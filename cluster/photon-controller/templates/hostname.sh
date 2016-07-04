#!/bin/bash

# Copyright 2014 The Kubernetes Authors.
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

# Remove kube.vm from /etc/hosts
sed -i -e 's/\b\w\+.vm\b//' /etc/hosts

# Update hostname in /etc/hosts and /etc/hostname
sed -i -e "s/\\bkube\\b/${MY_NAME}/g" /etc/host{s,name}
hostname ${MY_NAME}
