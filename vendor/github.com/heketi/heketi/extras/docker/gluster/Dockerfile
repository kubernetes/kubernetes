FROM centos

MAINTAINER Humble Chirammal hchiramm@redhat.com
LABEL version="0.5"
LABEL description="GlusterFS container based on CentOS 7"

ENV container docker

RUN yum --setopt=tsflags=nodocs -y update; yum clean all;

RUN (cd /lib/systemd/system/sysinit.target.wants/; for i in *; do [ $i == systemd-tmpfiles-setup.service ] || rm -f $i; done); \
rm -f /lib/systemd/system/multi-user.target.wants/*;\
rm -f /etc/systemd/system/*.wants/*;\
rm -f /lib/systemd/system/local-fs.target.wants/*; \
rm -f /lib/systemd/system/sockets.target.wants/*udev*; \
rm -f /lib/systemd/system/sockets.target.wants/*initctl*; \
rm -f /lib/systemd/system/basic.target.wants/*;\
rm -f /lib/systemd/system/anaconda.target.wants/*;

RUN yum --setopt=tsflags=nodocs -q -y install \
  wget \
  nfs-utils \
  attr \
  iputils \
  iproute \
  sudo \
  xfsprogs \
  centos-release-gluster \
  ntp \
  epel-release \
  openssh-clients \
  cronie \
  tar \
  rsync \
  sos ; yum clean all

RUN yum --setopt=tsflags=nodocs -y install \
  glusterfs \
  glusterfs-server \
  glusterfs-geo-replication ; yum clean all

# Backing up gluster config as it overlaps when bind mounting.
RUN mkdir -p /etc/glusterfs_bkp /var/lib/glusterd_bkp /var/log/glusterfs_bkp;\
cp -r /etc/glusterfs/* /etc/glusterfs_bkp;\
cp -r /var/lib/glusterd/* /var/lib/glusterd_bkp;\
cp -r /var/log/glusterfs/* /var/log/glusterfs_bkp;

# Adding script to move the glusterfs config file to location
ADD gluster-setup.service /etc/systemd/system/gluster-setup.service
RUN chmod 644 /etc/systemd/system/gluster-setup.service

# Adding script to move the glusterfs config file to location
ADD gluster-setup.sh /usr/sbin/gluster-setup.sh
RUN chmod 500 /usr/sbin/gluster-setup.sh

# To avoid the warnings while accessing the container
RUN sed -i "s/LANG/\#LANG/g" /etc/locale.conf

# Configure LVM so that we can create LVs and snapshots
RUN sed -i.save -e "s#udev_sync = 1#udev_sync = 0#" \
  -e "s#udev_rules = 1#udev_rules = 0#" \
  -e "s#use_lvmetad = 1#use_lvmetad = 0#" /etc/lvm/lvm.conf

# Set password
RUN echo 'root:password' | chpasswd

# Set SSH public key
USER root

VOLUME [ "/sys/fs/cgroup", "/dev", "/run/lvm" , "/var/lib/heketi" ]

EXPOSE 111 245 443 24007 2049 8080 6010 6011 6012 38465 38466 38468 38469 49152 49153 49154 49156 49157 49158 49159 49160 49161 49162

RUN systemctl disable nfs-server.service
RUN systemctl enable rpcbind.service
RUN systemctl enable ntpd.service
RUN systemctl enable gluster-setup.service
RUN systemctl enable glusterd.service

CMD ["/usr/sbin/init"]
