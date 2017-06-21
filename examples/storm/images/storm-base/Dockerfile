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

FROM fedora:21

MAINTAINER Matthew Farrellee <matt@cs.wisc.edu>

RUN yum update -y ; yum install -y java tar ; yum clean all

RUN curl -s http://archive.apache.org/dist/storm/apache-storm-0.9.3/apache-storm-0.9.3.tar.gz | \
      tar zxf - -C /opt && \
      mv /opt/apache-storm-0.9.3 /opt/apache-storm

ADD configure.sh /
