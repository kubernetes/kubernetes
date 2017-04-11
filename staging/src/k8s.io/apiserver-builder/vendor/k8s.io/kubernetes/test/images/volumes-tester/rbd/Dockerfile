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

# CEPH all in one
# Based on image by Ricardo Rocha, ricardo@catalyst.net.nz

FROM fedora

# Base Packages
RUN yum install -y wget ceph ceph-fuse strace && yum clean all

# Get ports exposed
EXPOSE 6789

ADD ./bootstrap.sh /bootstrap.sh
ADD ./mon.sh /mon.sh
ADD ./osd.sh /osd.sh
ADD ./ceph.conf.sh /ceph.conf.sh
ADD ./keyring /var/lib/ceph/mon/keyring
ADD ./block.tar.gz /

CMD /bootstrap.sh
