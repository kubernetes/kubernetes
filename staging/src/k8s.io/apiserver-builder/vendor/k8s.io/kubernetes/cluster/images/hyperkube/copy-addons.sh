#!/bin/bash
# Copyright 2016 The Kubernetes Authors.
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

# Now we're running in the sidecar container
# /etc/kubernetes/addons holds the data in the hyperkube container
# /srv/kubernetes is an emptyDir that maps to /etc/kubernetes in the addon-manager container
# This way we're using the latest manifests from hyperkube without updating
# kube-addon-manager which is used for other deployments too

# Inside /etc/kubernetes/addons, there are two directories: singlenode and multinode
# If "singlenode" is passed to this script; those manifests are copied to the addon-manager and vice versa with "multinode"
SUBDIRECTORY=$1

# While there is no data copied over to the emptyDir, try to copy it.
while [[ -z $(ls /srv/kubernetes/addons) ]]; do
	cp -r /etc/kubernetes/addons/${SUBDIRECTORY}/* /srv/kubernetes/addons/
done

# Then sleep forever
while true; do
	sleep 3600;
done
