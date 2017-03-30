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

FROM ubuntu:trusty

# add our user and group first to make sure their IDs get assigned 
# consistently, regardless of whatever dependencies get added
RUN groupadd -r mysql && useradd -r -g mysql mysql

ENV PERCONA_XTRADB_VERSION 5.6
ENV MYSQL_VERSION 5.6
ENV TERM linux

RUN apt-get update 
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y perl --no-install-recommends && rm -rf /var/lib/apt/lists/*

RUN apt-key adv --keyserver keys.gnupg.net --recv-keys 8507EFA5

RUN echo "deb http://repo.percona.com/apt trusty main" > /etc/apt/sources.list.d/percona.list
RUN echo "deb-src http://repo.percona.com/apt trusty main" >> /etc/apt/sources.list.d/percona.list

# the "/var/lib/mysql" stuff here is because the mysql-server
# postinst doesn't have an explicit way to disable the 
# mysql_install_db codepath besides having a database already
# "configured" (ie, stuff in /var/lib/mysql/mysql)
# also, we set debconf keys to make APT a little quieter
RUN { \
                echo percona-server-server-5.6 percona-server-server/data-dir select ''; \
                echo percona-server-server-5.6 percona-server-server/root_password password ''; \
        } | debconf-set-selections \
        && apt-get update && DEBIAN_FRONTEND=nointeractive apt-get install -y percona-xtradb-cluster-client-"${MYSQL_VERSION}" \ 
           percona-xtradb-cluster-common-"${MYSQL_VERSION}" percona-xtradb-cluster-server-"${MYSQL_VERSION}" \
        && rm -rf /var/lib/apt/lists/* \
        && rm -rf /var/lib/mysql && mkdir -p /var/lib/mysql && chown -R mysql:mysql /var/lib/mysql 

VOLUME /var/lib/mysql

COPY my.cnf /etc/mysql/my.cnf
COPY cluster.cnf /etc/mysql/conf.d/cluster.cnf

COPY docker-entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

EXPOSE 3306 4444 4567 4568
CMD ["mysqld"]
