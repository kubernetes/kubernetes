FROM fedora:21
MAINTAINER Jan Safranek <jsafrane@redhat.com>
EXPOSE 2049/tcp

RUN yum -y install nfs-utils && yum clean all

ADD run_nfs /usr/local/bin/

RUN chmod +x /usr/local/bin/run_nfs

ENTRYPOINT ["/usr/local/bin/run_nfs"]
