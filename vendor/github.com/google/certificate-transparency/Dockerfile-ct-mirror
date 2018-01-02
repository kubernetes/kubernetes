FROM ubuntu
RUN echo 'Building new CT Mirror Docker image...'
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    apt-add-repository -y ppa:jbboehr/coreos && \
    apt-get update && \
    apt-get install -qqy \
        ca-certificates \
        etcdctl
RUN groupadd -r ctlog && useradd -r -g ctlog ctlog
RUN mkdir /mnt/ctmirror
COPY cpp/server/ct-mirror /usr/local/bin/
COPY cpp/tools/ct-clustertool /usr/local/bin/
COPY cloud/keys /usr/local/etc/keys
VOLUME /mnt/ctmirror
CMD cd /mnt/ctmirror/ && \
    if [ ! -d logs ]; then mkdir logs; fi && \
    MY_IP=$(awk "/${HOSTNAME}/ {print \$1}" < /etc/hosts) && \
    export V_LEVEL=${V_LEVEL:-0} && \
    export NUM_HTTP_SERVER_THREADS=${NUM_HTTP_SERVER_THREADS:-32} && \
    echo "My IP: ${MY_IP}" && \
    echo "Container: ${CONTAINER_HOST}" && \
    echo "Etcd: ${ETCD_SERVERS}" && \
    echo "Target: ${TARGET_LOG_URL}" && \
    echo "Target TLS version: ${TARGET_LOG_TLS_VERSION}" && \
    echo "Target Key: ${TARGET_LOG_PUBLIC_KEY}" && \
    echo "Project: ${PROJECT}" && \
    echo "Monitoring: ${MONITORING}" && \
    ulimit -c unlimited && \
    /usr/local/bin/ct-mirror \
        --port=80 \
        --server=${CONTAINER_HOST} \
        --log_dir=/mnt/ctmirror/logs \
        --leveldb_db=/mnt/ctmirror/mirror.ldb \
        --etcd_servers="${ETCD_SERVERS}" \
        --num_http_server_threads=${NUM_HTTP_SERVER_THREADS} \
        --target_public_key=/usr/local/etc/keys/${TARGET_LOG_PUBLIC_KEY} \
        --target_log_uri=${TARGET_LOG_URL} \
        --monitoring=${MONITORING} \
        --google_compute_monitoring_base_url="https://www.googleapis.com/cloudmonitoring/v2beta2/projects/${PROJECT}" \
        --v=${V_LEVEL}; \
    if [ -e core ]; then \
      CORE_DIR="/mnt/ctmirror/cores/$(date +%s)"; \
      mkdir -p ${CORE_DIR}; \
      cp -v core ${CORE_DIR}; \
      cp -v /usr/local/bin/ct-mirror ${CORE_DIR}; \
      echo "Core saved to ${CORE_DIR}"; \
    fi

EXPOSE 80
