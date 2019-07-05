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

FROM BASEIMAGE as build_node_perf_npb_is

CROSS_BUILD_COPY qemu-QEMUARCH-static /usr/bin/

RUN apt-get update && apt-get install -y build-essential gfortran

ADD http://www.nas.nasa.gov/assets/npb/NPB3.3.1.tar.gz .
RUN tar xzf NPB3.3.1.tar.gz

WORKDIR ./NPB3.3.1/NPB3.3-OMP

# Create build config based on the architecture and build the workload.
RUN if [ $(arch) != "x86_64" ]; then \
    sed s/-mcmodel=medium//g config/NAS.samples/make.def.gcc_x86 > config/make.def; \
else \
    cp config/NAS.samples/make.def.gcc_x86 config/make.def; \
fi
RUN make IS CLASS=D

# Copying the required libraries (shared object files) to a convenient location so that it can be copied into the
# main container in the second build stage.
RUN mkdir -p /lib-copy && find /usr/lib -name "*.so.*" -exec cp {} /lib-copy \;

FROM BASEIMAGE

COPY --from=build_node_perf_npb_is /NPB3.3.1/NPB3.3-OMP/bin/is.D.x /
COPY --from=build_node_perf_npb_is /lib-copy /lib-copy
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/lib-copy"

ENTRYPOINT /is.D.x
