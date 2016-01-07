FROM ubuntu:14.04.3
MAINTAINER Mesosphere <support@mesosphere.io>

RUN locale-gen en_US.UTF-8
RUN dpkg-reconfigure locales
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8

RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -qqy \
        curl \
        openssl \
        && \
    apt-get clean

COPY ./bin/* /usr/local/bin/

ENTRYPOINT ["kube-keygen.sh"]
