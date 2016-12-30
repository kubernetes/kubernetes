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

FROM quay.io/pires/docker-jre:8u45-2


EXPOSE 5701

RUN \
  curl -Lskj https://github.com/pires/hazelcast-kubernetes-bootstrapper/releases/download/0.5/hazelcast-kubernetes-bootstrapper-0.5.jar \
  -o /bootstrapper.jar

CMD java -jar /bootstrapper.jar
