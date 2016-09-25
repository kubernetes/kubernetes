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

FROM node:0.10
MAINTAINER Christiaan Hees <christiaan@q42.nl>

ONBUILD WORKDIR /appsrc
ONBUILD COPY . /appsrc

ONBUILD RUN curl https://install.meteor.com/ | sh && \
    meteor build ../app --directory --architecture os.linux.x86_64 && \
    rm -rf /appsrc
# TODO rm meteor so it doesn't take space in the image?

ONBUILD WORKDIR /app/bundle

ONBUILD RUN (cd programs/server && npm install)
EXPOSE 8080
CMD []
ENV PORT 8080
ENTRYPOINT MONGO_URL=mongodb://$MONGO_SERVICE_HOST:$MONGO_SERVICE_PORT /usr/local/bin/node main.js
