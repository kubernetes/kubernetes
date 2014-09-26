# This Dockerfile will build an image that is configured
# to run fluentd with an elasticsearch plug-in and the
# provided configuration file.
# TODO(satnam): Use a lighter base image, e.g. some form of busybox.
# The image acts as an executable for the binary /usr/sbin/td-agent
# which runs fluentd with the default flag -v (which can be over-ridden).
# Note that fluentd is run with root permssion to allow access to
# log files with root only access under /var/lib/docker/containers/*

FROM ubuntu:14.04
MAINTAINER Satnam Singh "satnam@google.com"
RUN ulimit -n 65536
RUN apt-get update

RUN apt-get install -y curl
RUN apt-get install -y -q libcurl4-openssl-dev make
RUN /usr/bin/curl -L http://toolbelt.treasuredata.com/sh/install-ubuntu-trusty-td-agent2.sh | sh
RUN sed -i -e "s/USER=td-agent/USER=root/" -e "s/GROUP=td-agent/GROUP=root/" /etc/init.d/td-agent
RUN /usr/sbin/td-agent-gem install fluent-plugin-elasticsearch
COPY td-agent.conf /etc/td-agent/td-agent.conf

ENTRYPOINT ["/usr/sbin/td-agent"]
CMD ["-v"]
