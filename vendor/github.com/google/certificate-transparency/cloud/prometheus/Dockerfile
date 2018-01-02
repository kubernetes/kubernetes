FROM prom/prometheus
ENTRYPOINT echo "Config: ${PROMETHEUS_CONFIG}" && \
    cat ${PROMETHEUS_CONFIG} && \
    echo "Storage: ${PROMETHEUS_STORAGE}" && \
    /bin/prometheus \
      -logtostderr \
      -config.file=${PROMETHEUS_CONFIG} \
      -storage.local.path=${PROMETHEUS_STORAGE} \
      -web.console.libraries=/etc/prometheus/console_libraries \
      -web.console.templates=/etc/prometheus/consoles
CMD []
