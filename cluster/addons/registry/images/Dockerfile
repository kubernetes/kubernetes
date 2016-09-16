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

FROM haproxy:1.5
MAINTAINER Muhammed Uluyol <uluyol@google.com>

RUN apt-get update && apt-get install -y dnsutils

ADD proxy.conf.insecure.in /proxy.conf.in
ADD run_proxy.sh /usr/bin/run_proxy

RUN chown root:users /usr/bin/run_proxy
RUN chmod 755 /usr/bin/run_proxy

CMD ["/usr/bin/run_proxy"]
