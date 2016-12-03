FROM centos
MAINTAINER Humble Chirammal <hchiramm@redhat.com>

LABEL version="1.0"
LABEL description="Kubernetes Glusterfs DP test image"

RUN yum install heketi
ADD ./mock-heketi.json /etc/heketi/heketi.json

VOLUME /var/lib/heketi

# expose port, set user and set entrypoint with config option
ENTRYPOINT ["/usr/bin/heketi"]
EXPOSE 8080

CMD ["-config=/etc/heketi/heketi.json"]
