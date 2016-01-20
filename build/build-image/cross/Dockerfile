# Copyright 2014 Google Inc. All rights reserved.
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

# This file creates a standard build environment for building cross
# platform go binary for the architecture kubernetes cares about.

FROM  golang:1.5.2
# Original MAINTAINER Joe Beda <jbeda@google.com>
MAINTAINER  Brendan Burns <bburns@google.com>

ENV KUBE_CROSSPLATFORMS \
  linux/386 linux/arm \
  darwin/amd64 darwin/386 \
  windows/amd64 windows/386

# This is needed to compile the cross-compile stuff since golang is now self hosted
RUN curl -fsSL https://golang.org/dl/go1.4.2.linux-amd64.tar.gz -o golang.tar.gz \
         && tar -C /root -xzf golang.tar.gz \
         && mv /root/go /root/go1.4 \
         && rm golang.tar.gz

RUN cd /usr/local/go/src && for platform in ${KUBE_CROSSPLATFORMS}; do GOOS=${platform%/*} GOARCH=${platform##*/} ./make.bash --no-clean; done
