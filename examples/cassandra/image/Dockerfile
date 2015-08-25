FROM google/debian:wheezy

COPY cassandra.list /etc/apt/sources.list.d/cassandra.list

RUN gpg --keyserver pgp.mit.edu --recv-keys F758CE318D77295D
RUN gpg --export --armor F758CE318D77295D | apt-key add -

RUN gpg --keyserver pgp.mit.edu --recv-keys 2B5C1B00
RUN gpg --export --armor 2B5C1B00 | apt-key add -

RUN gpg --keyserver pgp.mit.edu --recv-keys 0353B12C
RUN gpg --export --armor 0353B12C | apt-key add -

RUN apt-get update
RUN apt-get -qq -y install procps cassandra

COPY cassandra.yaml /etc/cassandra/cassandra.yaml
COPY logback.xml /etc/cassandra/logback.xml
COPY run.sh /run.sh
COPY kubernetes-cassandra.jar /kubernetes-cassandra.jar

RUN chmod a+x /run.sh && \
    mkdir -p /cassandra_data/data && \
    chown -R cassandra.cassandra /etc/cassandra /cassandra_data && \
    chmod o+w -R /etc/cassandra /cassandra_data

VOLUME ["/cassandra_data/data"]    

USER cassandra

CMD /run.sh
