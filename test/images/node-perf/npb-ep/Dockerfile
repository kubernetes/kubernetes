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

FROM BASEIMAGE

RUN apt-get update && apt-get install -y build-essential gfortran

ADD http://www.nas.nasa.gov/assets/npb/NPB3.3.1.tar.gz .
RUN tar xzf NPB3.3.1.tar.gz

WORKDIR ./NPB3.3.1/NPB3.3-OMP
RUN cp config/NAS.samples/make.def.gcc_x86 config/make.def
RUN make EP CLASS=D

ENTRYPOINT ./bin/ep.D.x
