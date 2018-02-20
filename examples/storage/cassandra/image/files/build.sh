#!/bin/bash

# Copyright 2018 The Kubernetes Authors.
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


set -o errexit
set -o nounset
set -o pipefail

echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

apt-get update && apt-get dist-upgrade -y

clean-install \
    openjdk-8-jre-headless \
    libjemalloc1 \
    localepurge \
    dumb-init \
    wget

CASSANDRA_PATH="cassandra/${CASSANDRA_VERSION}/apache-cassandra-${CASSANDRA_VERSION}-bin.tar.gz"
CASSANDRA_DOWNLOAD="http://www.apache.org/dyn/closer.cgi?path=/${CASSANDRA_PATH}&as_json=1"
CASSANDRA_MIRROR=`wget -q -O - ${CASSANDRA_DOWNLOAD} | grep -oP "(?<=\"preferred\": \")[^\"]+"`

echo "Downloading Apache Cassandra from $CASSANDRA_MIRROR$CASSANDRA_PATH..."
wget -q -O - $CASSANDRA_MIRROR$CASSANDRA_PATH \
    | tar -xzf - -C /usr/local

mkdir -p /cassandra_data/data
mkdir -p /etc/cassandra

mv /logback.xml /cassandra.yaml /jvm.options /etc/cassandra/
mv /usr/local/apache-cassandra-${CASSANDRA_VERSION}/conf/cassandra-env.sh /etc/cassandra/

adduser --disabled-password --no-create-home --gecos '' --disabled-login cassandra
chmod +x /ready-probe.sh
chown cassandra: /ready-probe.sh

DEV_IMAGE=${DEV_CONTAINER:-}
if [ ! -z "$DEV_IMAGE" ]; then
    clean-install python;
else
    rm -rf  $CASSANDRA_HOME/pylib;
fi

apt-get -y purge localepurge
apt-get -y autoremove
apt-get clean

rm -rf \
    $CASSANDRA_HOME/*.txt \
    $CASSANDRA_HOME/doc \
    $CASSANDRA_HOME/javadoc \
    $CASSANDRA_HOME/tools/*.yaml \
    $CASSANDRA_HOME/tools/bin/*.bat \
    $CASSANDRA_HOME/bin/*.bat \
    doc \
    man \
    info \
    locale \
    common-licenses \
    ~/.bashrc \
    /var/lib/apt/lists/* \
    /var/log/* \
    /var/cache/debconf/* \
    /etc/systemd \
    /lib/lsb \
    /lib/udev \
    /usr/share/doc/ \
    /usr/share/doc-base/ \
    /usr/share/man/ \
    /tmp/* \
    /usr/lib/jvm/java-8-openjdk-amd64/jre/plugin \
    /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/javaws \
    /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/jjs \
    /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/orbd \
    /usr/lib/jvm/java-8-openjdk-amd64/bin/pack200 \
    /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/policytool \
    /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/rmid \
    /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/rmiregistry \
    /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/servertool \
    /usr/lib/jvm/java-8-openjdk-amd64/bin/tnameserv \
    /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/unpack200 \
    /usr/lib/jvm/java-8-openjdk-amd64/jre/lib/javaws.jar \
    /usr/lib/jvm/java-8-openjdk-amd64/jre/lib/deploy* \
    /usr/lib/jvm/java-8-openjdk-amd64/jre/lib/desktop \
    /usr/lib/jvm/java-8-openjdk-amd64/jre/lib/*javafx* \
    /usr/lib/jvm/java-8-openjdk-amd64/jre/lib/*jfx* \
    /usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/libdecora_sse.so \
    /usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/libprism_*.so \
    /usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/libfxplugins.so \
    /usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/libglass.so \
    /usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/libgstreamer-lite.so \
    /usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/libjavafx*.so \
    /usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/libjfx*.so \
    /usr/lib/jvm/java-8-openjdk-amd64/jre/lib/ext/jfxrt.jar \
    /usr/lib/jvm/java-8-openjdk-amd64/jre/lib/ext/nashorn.jar \
    /usr/lib/jvm/java-8-openjdk-amd64/jre/lib/oblique-fonts \
    /usr/lib/jvm/java-8-openjdk-amd64/jre/lib/plugin.jar \
    /usr/lib/jvm/java-8-openjdk-amd64/man
