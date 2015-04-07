FROM google/debian:wheezy

COPY backports.list /etc/apt/sources.list.d/backports.list

RUN apt-get update
RUN apt-get -t wheezy-backports -yy -q install nginx
