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

FROM fedora:24
LABEL maintainers="Jan Safranek, jsafrane@redhat.com; Matthew Wong, mawong@redhat.com"

# install nfs-ganesha
RUN dnf install -y tar gcc cmake autoconf libtool bison flex make gcc-c++ krb5-devel dbus-devel && dnf clean all \
    && curl -L https://github.com/nfs-ganesha/nfs-ganesha/archive/V2.4.0.3.tar.gz | tar zx \
    && curl -L https://github.com/nfs-ganesha/ntirpc/archive/v1.4.1.tar.gz | tar zx \
    && rm -r nfs-ganesha-2.4.0.3/src/libntirpc \
    && mv ntirpc-1.4.1 nfs-ganesha-2.4.0.3/src/libntirpc \
    && cd nfs-ganesha-2.4.0.3 \
    && sed -i '20i#cmakedefine _NO_PORTMAPPER 1' src/include/config-h.in.cmake \
    && cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_CONFIG=vfs_only -D_NO_PORTMAPPER=ON src/ \
    && make \
    && make install \
    && dnf remove -y tar gcc cmake autoconf libtool bison flex make gcc-c++ krb5-devel dbus-devel && dnf clean all

# also need this for ganesha to run
RUN dnf install -y libntirpc

RUN mkdir -p /exports

# expose mountd 20048/tcp and nfsd 2049/tcp
EXPOSE 2049/tcp 20048/tcp

COPY index.html /tmp/index.html
RUN chmod 644 /tmp/index.html
COPY run_nfs.sh /usr/local/bin/

ENTRYPOINT ["/usr/local/bin/run_nfs.sh"]
CMD ["/exports"]
