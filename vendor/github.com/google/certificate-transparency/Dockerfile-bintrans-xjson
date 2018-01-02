FROM ubuntu
RUN echo 'Building new SuperDuper XJSON Docker image...'
COPY test/testdata/ca-cert.pem /tmp/
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    apt-add-repository -y ppa:jbboehr/coreos && \
    apt-get update && \
    apt-get install -qqy \
        ca-certificates \
        etcdctl
RUN groupadd -r ctlog && useradd -r -g ctlog ctlog
RUN mkdir /mnt/ctlog
COPY cpp/server/xjson-server /usr/local/bin/
COPY test/testdata/ct-server-key.pem /usr/local/etc/server-key.pem
COPY cpp/tools/ct-clustertool /usr/local/bin/
VOLUME /mnt/ctlog
CMD cd /mnt/ctlog/ && \
    if [ ! -d logs ]; then mkdir logs; fi && \
    MY_IP=$(awk "/${HOSTNAME}/ {print \$1}" < /etc/hosts) && \
    export V_LEVEL=${V_LEVEL:-0} && \
    export NUM_HTTP_SERVER_THREADS=${NUM_HTTP_SERVER_THREADS:-32} && \
    echo "My IP: ${MY_IP}" && \
    echo "Container: ${CONTAINER_HOST}" && \
    echo "Etcd: ${ETCD_SERVERS}" && \
    echo "Project: ${PROJECT}" && \
    echo "Monitoring: ${MONITORING}" && \
    ulimit -c unlimited && \
    /usr/local/bin/xjson-server \
        --port=80 \
        --server=${CONTAINER_HOST} \
        --key=/usr/local/etc/server-key.pem \
        --log_dir=/mnt/ctlog/logs \
        --tree_signing_frequency_seconds=30 \
        --guard_window_seconds=10 \
        --leveldb_db=/mnt/ctlog/log.ldb \
        --etcd_servers="${ETCD_SERVERS}" \
        --etcd_delete_concurrency=100 \
        --num_http_server_threads=${NUM_HTTP_SERVER_THREADS} \
        --monitoring=${MONITORING} \
        --google_compute_monitoring_base_url="https://www.googleapis.com/cloudmonitoring/v2beta2/projects/${PROJECT}" \
        --v=${V_LEVEL}; \
    if [ -e core ]; then \
      CORE_DIR="/mnt/ctlog/cores/$(date +%s)"; \
      mkdir -p ${CORE_DIR}; \
      cp -v core ${CORE_DIR}; \
      cp -v /usr/local/bin/xjson-server ${CORE_DIR}; \
      echo "Core saved to ${CORE_DIR}"; \
    fi

EXPOSE 80
