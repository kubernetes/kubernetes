# Copyright 2018 The Kubernetes Authors.
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

ARG BASEIMAGE
FROM $BASEIMAGE as build_node_perf_npb_ep

CROSS_BUILD_COPY qemu-QEMUARCH-static /usr/bin/

RUN rm -f /etc/apt/sources.list
RUN echo "deb [trusted=yes] http://archive.debian.org/debian buster main" >> /etc/apt/sources.list
RUN echo "deb-src [trusted=yes] http://archive.debian.org/debian buster main" >> /etc/apt/sources.list
RUN echo "deb [trusted=yes] http://archive.debian.org/debian buster-updates main" >> /etc/apt/sources.list
RUN echo "deb-src [trusted=yes] http://archive.debian.org/debian buster-updates main" >> /etc/apt/sources.list

RUN apt-get update && apt-get install -y --allow-downgrades libbz2-1.0=1.0.6-9.2~deb10u1 libc6=2.28-10+deb10u1 libc6-dev g++ bzip2 dpkg-dev gfortran-8 build-essential gfortran

ADD http://www.nas.nasa.gov/assets/npb/NPB3.4.3.tar.gz .
RUN tar xzf NPB3.4.3.tar.gz

WORKDIR ./NPB3.4.3/NPB3.4-OMP
RUN if [ $(arch) != "x86_64" ]; then \
    sed s/-mcmodel=medium//g config/NAS.samples/make.def_gcc > config/make.def; \
else \
    cp config/NAS.samples/make.def_gcc config/make.def; \
fi
RUN make EP CLASS=D

# Copying the required libraries (shared object files) to a convenient location so that it can be copied into the
# main container in the second build stage.
RUN mkdir -p /lib-copy && find /usr/lib -name "*.so.*" -exec cp {} /lib-copy \;

FROM $BASEIMAGE

COPY --from=build_node_perf_npb_ep /NPB3.4.3/NPB3.4-OMP/bin/ep.D.x /
COPY --from=build_node_perf_npb_ep /lib-copy /lib-copy
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/lib-copy"

ENTRYPOINT /ep.D.x
