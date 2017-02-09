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

FROM gcr.io/google_containers/ubuntu-slim:0.6
RUN apt-get update && apt-get install -y --no-install-recommends iperf bash \
  && apt-get clean -y \
  && rm -rf /var/lib/apt/lists/* \
  && ln -s /usr/bin/iperf /usr/local/bin/iperf
RUN ls -altrh /usr/local/bin/iperf
