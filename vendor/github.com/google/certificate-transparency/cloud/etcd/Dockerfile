FROM ubuntu
RUN \
 sed 's/main$/main universe/' -i /etc/apt/sources.list && \
 apt-get update && \
 env DEBIAN_FRONTEND=noninteractive apt-get install -y curl
RUN \
 cd /tmp && \
 (curl -L  https://github.com/coreos/etcd/releases/download/v2.0.9/etcd-v2.0.9-linux-amd64.tar.gz | tar -xz) && \
 mkdir -p /opt/etcd/bin && \
 cp -v /tmp/etcd-v2.0.9-linux-amd64/etcd /opt/etcd/bin && \
 cp -v /tmp/etcd-v2.0.9-linux-amd64/etcdctl /opt/etcd/bin && \
 rm -rf /tmp/etcd-v2.0.9-linux-amd64
WORKDIR /opt/etcd
VOLUME ["/opt/etcd/data"]
CMD MY_IP=$(awk "/${HOSTNAME}/ {print \$1}" < /etc/hosts) && \
    export HEARTBEAT_INTERVAL=${HEARTBEAT_INTERVAL:-100} && \
    echo "My IP: ${MY_IP}" && \
    echo "Container host: ${CONTAINER_HOST}" && \
    echo "My Discovery: ${DISCOVERY}" && \
    echo "Heartbeat interval: ${HEARTBEAT_INTERVAL}" && \
    /opt/etcd/bin/etcd --discovery="${DISCOVERY}" \
          --name=${ETCD_NAME} \
          --advertise-client-urls=http://${CONTAINER_HOST}:4001 \
          --initial-advertise-peer-urls=http://${CONTAINER_HOST}:7001 \
          --listen-client-urls=http://${MY_IP}:4001 \
          --listen-peer-urls=http://${MY_IP}:7001 \
          --data-dir=/opt/etcd/data \
          --heartbeat-interval=${HEARTBEAT_INTERVAL} \
          --election-timeout=6000
EXPOSE 4001 7001

