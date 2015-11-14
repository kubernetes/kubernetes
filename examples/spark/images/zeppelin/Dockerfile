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

# Based heavily on
# https://github.com/dylanmei/docker-zeppelin/blob/master/Dockerfile
# (which is similar to many others out there), but rebased onto maven
# image.
#
# This image is a composition of the official docker-maven
# Docker image from https://github.com/carlossg/docker-maven/ and
# spark-base.

FROM gcr.io/google_containers/spark-base:latest

ENV ZEPPELIN_TAG  v0.5.5
ENV MAVEN_VERSION 3.3.3
ENV SPARK_MINOR   1.5
ENV SPARK_PATCH   1
ENV SPARK_VER     ${SPARK_MINOR}.${SPARK_PATCH}
ENV HADOOP_MINOR  2.6
ENV HADOOP_PATCH  1
ENV HADOOP_VER    ${HADOOP_MINOR}.${HADOOP_PATCH}

RUN curl -fsSL http://archive.apache.org/dist/maven/maven-3/${MAVEN_VERSION}/binaries/apache-maven-${MAVEN_VERSION}-bin.tar.gz | tar xzf - -C /usr/share \
  && mv /usr/share/apache-maven-${MAVEN_VERSION} /usr/share/maven \
  && ln -s /usr/share/maven/bin/mvn /usr/bin/mvn

ENV MAVEN_HOME /usr/share/maven

# libfontconfig is a workaround for
# https://github.com/karma-runner/karma/issues/1270, which caused a
# build break similar to
# https://www.mail-archive.com/users@zeppelin.incubator.apache.org/msg01586.html

RUN apt-get update \
  && apt-get install -y net-tools build-essential git wget unzip python python-setuptools python-dev python-numpy libfontconfig \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/apache/incubator-zeppelin.git --branch ${ZEPPELIN_TAG} /opt/zeppelin
RUN cd /opt/zeppelin && \
  mvn clean package \
    -Pspark-${SPARK_MINOR} -Dspark.version=${SPARK_VER} \
    -Phadoop-${HADOOP_MINOR} -Dhadoop.version=${HADOOP_VER} \
    -Ppyspark \
    -DskipTests && \
  rm -rf /root/.m2 && \
  rm -rf /root/.npm && \
  echo "Successfully built Zeppelin"

ADD zeppelin-log4j.properties /opt/zeppelin/conf/log4j.properties
ADD zeppelin-env.sh /opt/zeppelin/conf/zeppelin-env.sh
ADD docker-zeppelin.sh /opt/zeppelin/bin/docker-zeppelin.sh
EXPOSE 8080
ENTRYPOINT ["/opt/zeppelin/bin/docker-zeppelin.sh"]
