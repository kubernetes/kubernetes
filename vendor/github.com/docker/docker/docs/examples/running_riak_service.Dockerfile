# Riak
#
# VERSION       0.1.1

# Use the Ubuntu base image provided by dotCloud
FROM ubuntu:trusty
MAINTAINER Hector Castro hector@basho.com

# Install Riak repository before we do apt-get update, so that update happens
# in a single step
RUN apt-get install -q -y curl && \
    curl -sSL https://packagecloud.io/install/repositories/basho/riak/script.deb | sudo bash

# Install and setup project dependencies
RUN apt-get update && \
    apt-get install -y supervisor riak=2.0.5-1

RUN mkdir -p /var/log/supervisor

RUN locale-gen en_US en_US.UTF-8

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Configure Riak to accept connections from any host
RUN sed -i "s|listener.http.internal = 127.0.0.1:8098|listener.http.internal = 0.0.0.0:8098|" /etc/riak/riak.conf
RUN sed -i "s|listener.protobuf.internal = 127.0.0.1:8087|listener.protobuf.internal = 0.0.0.0:8087|" /etc/riak/riak.conf

# Expose Riak Protocol Buffers and HTTP interfaces
EXPOSE 8087 8098

CMD ["/usr/bin/supervisord"]
