# Copyright 2016 The Kubernetes Authors All rights reserved.
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

# This docker image runs Jenkins Job Builder (JJB) for automatic job reconciliation.

FROM ubuntu:14.04
MAINTAINER Joe Finney <spxtr@google.com>

RUN mkdir /build
WORKDIR /build

# Dependencies for JJB
RUN apt-get update && apt-get install -y \
    wget \
    git \
    python-dev \
    python-pip \
    libyaml-dev \
    python-yaml
RUN pip install PyYAML python-jenkins
# Required since JJB supports python 2.6, which doesn't have ordereddict built-in. We have 2.7.
RUN wget https://pypi.python.org/packages/source/o/ordereddict/ordereddict-1.1.tar.gz \
    && tar -xvf ordereddict-1.1.tar.gz \
    && cd ordereddict-1.1 \
    && python setup.py install

RUN git clone https://git.openstack.org/openstack-infra/jenkins-job-builder \
    && cd jenkins-job-builder \
    && python setup.py install

# JJB configuration lives in /etc/jenkins_jobs/jenkins_jobs.ini
RUN mkdir -p /etc/jenkins_jobs

WORKDIR /
RUN git clone https://github.com/kubernetes/kubernetes.git
WORKDIR kubernetes
