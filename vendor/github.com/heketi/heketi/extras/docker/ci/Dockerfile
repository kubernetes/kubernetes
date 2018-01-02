# set author and base
FROM centos
MAINTAINER Luis Pabón <lpabon@redhat.com>

LABEL version="1.0"
LABEL description="CI Development build"

# post install config and volume setup
ADD ./heketi /usr/bin/heketi
ADD ./heketi-start.sh /usr/bin/heketi-start.sh
ADD ./heketi.json /etc/heketi/heketi.json

RUN mkdir /var/lib/heketi
VOLUME /var/lib/heketi

# expose port, set user and set entrypoint with config option
ENTRYPOINT ["/usr/bin/heketi-start.sh"]
EXPOSE 8080
