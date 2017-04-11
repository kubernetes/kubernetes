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

FROM centos
RUN yum -y install hostname centos-release-gluster && yum -y install glusterfs-server && yum clean all
ADD glusterd.vol /etc/glusterfs/
ADD run_gluster.sh /usr/local/bin/
ADD index.html /vol/
RUN chmod 644 /vol/index.html

EXPOSE 24007/tcp 49152/tcp

ENTRYPOINT ["/usr/local/bin/run_gluster.sh"]
