#!/bin/bash

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

yum update -y -v
yum install   openssh openssh-server openssh-clients hostname -y -q

ssh-keygen -f ~/.ssh/id_rsa -t rsa -N ''
cat ~/.ssh/id_rsa.pub |awk '{print $1, $2, "Generated"}' >> ~/.ssh/authorized_keys2
cat ~/.ssh/id_rsa.pub |awk '{print $1, $2, "Generated"}' >> ~/.ssh/authorized_keys

rpm -Uvh http://ceph.com/rpm/rhel6/noarch/ceph-release-1-0.el6.noarch.rpm
yum install  -y -q python-itsdangerous python-werkzeug python-jinja2 python-flask ceph-deploy epel-release
# ceph pkg depends on epel-release
yum install -y -q  ceph ceph-fuse
