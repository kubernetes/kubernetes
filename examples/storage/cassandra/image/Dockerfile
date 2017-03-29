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

FROM gcr.io/google_containers/ubuntu-slim:0.6

ARG BUILD_DATE
ARG VCS_REF
ARG CASSANDRA_VERSION
ARG DEV_CONTAINER

LABEL \
    org.label-schema.build-date=$BUILD_DATE \
    org.label-schema.docker.dockerfile="/Dockerfile" \
    org.label-schema.license="Apache License 2.0" \
    org.label-schema.name="k8s-for-greeks/docker-cassandra-k8s" \
    org.label-schema.url="https://github.com/k8s-for-greeks/" \
    org.label-schema.vcs-ref=$VCS_REF \
    org.label-schema.vcs-type="Git" \
    org.label-schema.vcs-url="https://github.com/k8s-for-greeks/docker-cassandra-k8s"

ENV CASSANDRA_HOME=/usr/local/apache-cassandra-${CASSANDRA_VERSION} \
    CASSANDRA_CONF=/etc/cassandra \
    CASSANDRA_DATA=/cassandra_data \
    CASSANDRA_LOGS=/var/log/cassandra \
    JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64 \
    PATH=${PATH}:/usr/lib/jvm/java-8-openjdk-amd64/bin:/usr/local/apache-cassandra-${CASSANDRA_VERSION}/bin  \
    DI_VERSION=1.2.0 \
    DI_SHA=81231da1cd074fdc81af62789fead8641ef3f24b6b07366a1c34e5b059faf363

ADD files /

RUN set -e && echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections \
  && apt-get update && apt-get -qq -y --force-yes install --no-install-recommends \
	openjdk-8-jre-headless \
	libjemalloc1 \
	localepurge \
	wget && \
  mirror_url=$( wget -q -O - http://www.apache.org/dyn/closer.cgi/cassandra/ \
        | sed -n 's#.*href="\(http://ftp.[^"]*\)".*#\1#p' \
        | head -n 1 \
    ) \
    && wget -q -O - ${mirror_url}/${CASSANDRA_VERSION}/apache-cassandra-${CASSANDRA_VERSION}-bin.tar.gz \
        | tar -xzf - -C /usr/local \
    && wget -q -O - https://github.com/Yelp/dumb-init/releases/download/v${DI_VERSION}/dumb-init_${DI_VERSION}_amd64 > /sbin/dumb-init \
    && echo "$DI_SHA  /sbin/dumb-init" | sha256sum -c - \
    && chmod +x /sbin/dumb-init \
    && chmod +x /ready-probe.sh \
    && mkdir -p /cassandra_data/data \
    && mkdir -p /etc/cassandra \
    && mv /logback.xml /cassandra.yaml /jvm.options /etc/cassandra/ \
    && mv /usr/local/apache-cassandra-${CASSANDRA_VERSION}/conf/cassandra-env.sh /etc/cassandra/ \
    && adduser --disabled-password --no-create-home --gecos '' --disabled-login cassandra \
    && chown cassandra: /ready-probe.sh \
    && if [ -n "$DEV_CONTAINER" ]; then apt-get -y --no-install-recommends install python; else rm -rf  $CASSANDRA_HOME/pylib; fi \
    && apt-get -y purge wget localepurge \
    && apt-get autoremove \
    && apt-get clean \
    && rm -rf \
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

VOLUME ["/$CASSANDRA_DATA"]

# 7000: intra-node communication
# 7001: TLS intra-node communication
# 7199: JMX
# 9042: CQL
# 9160: thrift service
EXPOSE 7000 7001 7199 9042 9160

CMD ["/sbin/dumb-init", "/bin/bash", "/run.sh"]
