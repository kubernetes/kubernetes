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

FROM BASEIMAGE

# Create symlinks for each hyperkube server
# Also create symlinks to /usr/local/bin/ where the server image binaries live, so the hyperkube image may be
# used instead of gcr.io/google_containers/kube-* without any modifications.
# TODO: replace manual symlink creation with --make-symlink command once
# cross-building with qemu supports go binaries. See #28702
# RUN /hyperkube --make-symlinks
RUN ln -s /hyperkube /apiserver \
 && ln -s /hyperkube /controller-manager \
 && ln -s /hyperkube /kubectl \
 && ln -s /hyperkube /kubelet \
 && ln -s /hyperkube /proxy \
 && ln -s /hyperkube /scheduler \
  && ln -s /hyperkube /aggerator \
 && ln -s /hyperkube /usr/local/bin/kube-apiserver \
 && ln -s /hyperkube /usr/local/bin/kube-controller-manager \
 && ln -s /hyperkube /usr/local/bin/kubectl \
 && ln -s /hyperkube /usr/local/bin/kubelet \
 && ln -s /hyperkube /usr/local/bin/kube-proxy \
 && ln -s /hyperkube /usr/local/bin/kube-scheduler

# Copy the hyperkube binary
COPY hyperkube /hyperkube
