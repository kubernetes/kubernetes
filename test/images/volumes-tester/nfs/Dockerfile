# Copyright 2015 Google Inc. All rights reserved.
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
MAINTAINER Jan Safranek, jsafrane@redhat.com
RUN yum -y install /usr/bin/ps nfs-utils && yum clean all
RUN mkdir -p /exports
ADD run_nfs.sh /usr/local/bin/
ADD index.html /exports/index.html
RUN chmod 644 /exports/index.html

EXPOSE 2049/tcp

ENTRYPOINT ["/usr/local/bin/run_nfs.sh", "/exports"]
