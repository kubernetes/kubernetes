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

# attempt to warn user about kube and etcd binaries
PATH=$PATH:/opt/bin:

if ! $(grep Ubuntu /etc/lsb-release > /dev/null 2>&1)
then
    echo "warning: not detecting a ubuntu system"
fi

if ! $(which etcd > /dev/null)
then
    echo "warning: etcd binary is not found in the PATH: $PATH"
fi

if ! $(which kube-apiserver > /dev/null) && ! $(which kubelet > /dev/null)
then
    echo "warning: kube binaries are not found in the $PATH"
fi

# copy /etc/init files
cp init_conf/* /etc/init/

# copy /etc/initd/ files
cp initd_scripts/* /etc/init.d/

# copy default configs
cp default_scripts/* /etc/default/

