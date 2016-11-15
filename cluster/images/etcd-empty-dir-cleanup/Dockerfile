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

FROM gliderlabs/alpine
MAINTAINER Mehdy Bohlool <mehdy@google.com>

RUN apk-install bash
ADD etcd-empty-dir-cleanup.sh etcd-empty-dir-cleanup.sh
ADD etcdctl etcdctl
ENV ETCDCTL /etcdctl
ENV SLEEP_SECOND 3600
RUN chmod +x etcd-empty-dir-cleanup.sh
CMD bash /etcd-empty-dir-cleanup.sh
