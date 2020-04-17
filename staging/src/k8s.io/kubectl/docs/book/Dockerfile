# Copyright 2019 The Kubernetes Authors.
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

FROM python:3.7

EXPOSE 4000
RUN curl -sL https://deb.nodesource.com/setup_11.x | bash

RUN apt-get update && apt-get install -y nodejs npm && apt-get clean;
RUN npm install gitbook-cli -g

WORKDIR /opt/book/
COPY . /opt/book/
RUN npm install

CMD ["gitbook", "serve"]
