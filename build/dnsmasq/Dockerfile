# Copyright 2016 The Kubernetes Authors. All rights reserved.
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

FROM alpine:3.4
MAINTAINER Prashanth.B <beeps@google.com>
RUN apk --no-cache add dnsmasq
COPY dnsmasq.conf /etc/dnsmasq.conf
COPY babysitter /babysitter
RUN chmod -c 777 /babysitter
# Replace SIGTERM with SIGCONT as stop signal to support graceful termination
STOPSIGNAL SIGCONT
ENTRYPOINT ["/babysitter"]
# ENTRYPOINT ["/usr/sbin/dnsmasq", "--keep-in-foreground"]
