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

FROM ubuntu:focal

# ubuntu focal maintained glusterfs
RUN apt-get update && apt-get install -y netbase nfs-common=1:1.3.4-2.5ubuntu3.3 glusterfs-client=7.2-2build1

# official distribution
# RUN apt-get update && apt-get install software-properties-common -y && add-apt-repository ppa:gluster/glusterfs-8 && apt-get update && apt-get install glusterfs-client -y

ENTRYPOINT ["/bin/mount"]
