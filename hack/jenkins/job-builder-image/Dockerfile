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
