FROM mesosphere/mesos-slave-dind:mesos-0.24.0_dind-0.2_docker-1.8.2_ubuntu-14.04.3
MAINTAINER Mesosphere <support@mesosphere.io>

RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -qqy socat && \
    apt-get clean

COPY ./downloads/* /usr/local/bin/
