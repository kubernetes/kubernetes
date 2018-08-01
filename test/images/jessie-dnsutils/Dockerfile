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

FROM BASEIMAGE

CROSS_BUILD_COPY qemu-QEMUARCH-static /usr/bin/

# WARNING: Please note that the script below removes the security packages from arm64 and ppc64el images
# as they do not exist anymore in the debian repositories for jessie. So we do not recommend using this
# image for any production use and limit use of this image to just test scenarios.

COPY fixup-apt-list.sh /
RUN ["/fixup-apt-list.sh"]

RUN apt-get -q update && \
    apt-get install -y dnsutils && \
    apt-get clean
