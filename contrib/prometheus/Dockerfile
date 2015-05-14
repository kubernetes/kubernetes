FROM       prom/prometheus
MAINTAINER Marcin Wielgus <mwielgus@google.com>

COPY ./run_prometheus.sh /prometheus/run_prometheus.sh

ENTRYPOINT ["/bin/sh", "/prometheus/run_prometheus.sh"]
CMD []
