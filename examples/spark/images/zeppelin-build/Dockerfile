# Copyright 2015 The Kubernetes Authors All rights reserved.
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

# This is the Zeppelin *build* image. It spits out a /zeppelin.tgz
# alone, which is then copied out by the Makefile and used in the
# actual Zeppelin image.
#
# Based heavily on
# https://github.com/dylanmei/docker-zeppelin/blob/master/Dockerfile
# (which is similar to many others out there), but rebased onto maven
# image.

FROM maven:3.3.3-jdk-8

ENV ZEPPELIN_TAG  v0.5.5
ENV SPARK_MINOR   1.5
ENV SPARK_PATCH   1
ENV SPARK_VER     ${SPARK_MINOR}.${SPARK_PATCH}
ENV HADOOP_MINOR  2.6
ENV HADOOP_PATCH  1
ENV HADOOP_VER    ${HADOOP_MINOR}.${HADOOP_PATCH}

# libfontconfig is a workaround for
# https://github.com/karma-runner/karma/issues/1270, which caused a
# build break similar to
# https://www.mail-archive.com/users@zeppelin.incubator.apache.org/msg01586.html

RUN apt-get update \
  && apt-get install -y net-tools build-essential git wget unzip python python-setuptools python-dev python-numpy libfontconfig

RUN git clone https://github.com/apache/incubator-zeppelin.git --branch ${ZEPPELIN_TAG} /opt/zeppelin
RUN cd /opt/zeppelin && \
  mvn clean package \
    -Pbuild-distr \
    -Pspark-${SPARK_MINOR} -Dspark.version=${SPARK_VER} \
    -Phadoop-${HADOOP_MINOR} -Dhadoop.version=${HADOOP_VER} \
    -Ppyspark \
    -DskipTests && \
  echo "Successfully built Zeppelin"

RUN cd /opt/zeppelin/zeppelin-distribution/target/zeppelin-* && \
  mv zeppelin-* zeppelin && \
  tar cvzf /zeppelin.tgz zeppelin
