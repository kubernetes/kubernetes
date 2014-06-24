FROM       google/kubernetes
MAINTAINER https://github.com/GoogleCloudPlatform/kubernetes

ENV  ETCD_VERSION 0.4.3
RUN  mkdir /etcd && touch /etc/kubelet.conf
RUN  curl -Ls https://github.com/coreos/etcd/releases/download/v${ETCD_VERSION}/etcd-v${ETCD_VERSION}-linux-amd64.tar.gz | tar -C /etcd --strip-components=1 -xvzf -

ADD  run.sh    /
RUN  chmod a+x /run.sh

EXPOSE 10250 4001 7001
ENTRYPOINT [ "/run.sh" ]
