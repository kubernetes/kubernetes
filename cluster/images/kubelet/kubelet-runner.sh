#!/bin/bash

# Copyright 2015 The Kubernetes Authors All rights reserved.
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

# This will run kubelet outside of the container (in host namespaces) and will
# exit when kubelet exits with the same error code.
# TODO: This script is used to workaround lack of namespace propagation flag in docker
#       (see https://github.com/docker/docker/issues/14630). This got fixed in PR
#       https://github.com/docker/docker/pull/17034 and will be released in docker 1.10.
#       As long as we support docker versions older than 1.10 we should run kubelet
#       via nsenter.
nsenter --target=1 --mount --wd=. -- ./kubelet $@
