FROM google/debian:wheezy

COPY cassandra.list /etc/apt/sources.list.d/cassandra.list

RUN gpg --keyserver pgp.mit.edu --recv-keys F758CE318D77295D
RUN gpg --export --armor F758CE318D77295D | apt-key add -

RUN gpg --keyserver pgp.mit.edu --recv-keys 2B5C1B00
RUN gpg --export --armor 2B5C1B00 | apt-key add -

RUN gpg --keyserver pgp.mit.edu --recv-keys 0353B12C
RUN gpg --export --armor 0353B12C | apt-key add -

RUN apt-get update
RUN apt-get -qq -y install cassandra

COPY cassandra.yaml /etc/cassandra/cassandra.yaml
COPY run.sh /run.sh
COPY kubernetes-cassandra.jar /kubernetes-cassandra.jar
RUN chmod a+x /run.sh

CMD /run.sh
