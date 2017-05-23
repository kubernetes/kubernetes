# Copyright 2017 The Kubernetes Authors.
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

FROM ubuntu:16.04

# Disable prompts from apt
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get -qq update
RUN apt-get install -qq pciutils gcc g++ git make dpkg-dev bc module-init-tools curl 

RUN mkdir /lakitu-kernel
RUN git clone https://chromium.googlesource.com/chromiumos/third_party/kernel /lakitu-kernel

ADD installer.sh /usr/bin/nvidia-installer.sh
RUN chmod a+x /usr/bin/nvidia-installer.sh
CMD ["/usr/bin/nvidia-installer.sh"]