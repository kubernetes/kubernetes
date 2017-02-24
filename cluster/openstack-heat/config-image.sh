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

## Contains configuration values for new image. It is skip when CREATE_IMAGE=false

# Image name which will be displayed in OpenStack
OPENSTACK_IMAGE_NAME=${OPENSTACK_IMAGE_NAME:-CentOS-7-x86_64-GenericCloud-1604}

# Downloaded image name for Openstack project
IMAGE_FILE=${IMAGE_FILE:-CentOS-7-x86_64-GenericCloud-1604.qcow2}

# Absolute path where image file is stored.
IMAGE_PATH=${IMAGE_PATH:-~/Downloads/openstack}

# The URL basepath for downloading the image
IMAGE_URL_PATH=${IMAGE_URL_PATH:-http://cloud.centos.org/centos/7/images}

# The disk format of the image. Acceptable formats are ami, ari, aki, vhd, vmdk, raw, qcow2, vdi, and iso.
IMAGE_FORMAT=${IMAGE_FORMAT:-qcow2}

# The container format of the image. Acceptable formats are ami, ari, aki, bare, docker, and ovf.
CONTAINER_FORMAT=${CONTAINER_FORMAT:-bare}
