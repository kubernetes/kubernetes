#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/kube-config/influxdb"

start() {
  if kubectl.sh create -f "$DIR/" &> /dev/null; then
    echo "heapster pods have been setup"
  else 
    echo "failed to setup heapster pods"
  fi
}

stop() {
  kubectl.sh stop replicationController monitoring-influx-grafana-controller &> /dev/null
  kubectl.sh stop replicationController monitoring-heapster-controller &> /dev/null
  # wait for the pods to disappear.
  while kubectl.sh get pods -l "name=influxGrafana" -o template -t {{range.items}}{{.id}}:{{end}} | grep -c . &> /dev/null \
    || kubectl.sh get pods -l "name=heapster" -o template -t {{range.items}}{{.id}}:{{end}} | grep -c . &> /dev/null; do
    sleep 2
  done
  kubectl.sh delete -f "$DIR/" &> /dev/null || true
  echo "heapster pods have been removed."
}

case "$1" in
  start)
    start
    ;;
  stop)
    stop
    ;;
  restart)
    stop
    start
    ;;
  *)
    echo "Usage: $0 {start|stop|restart}"
    ;;
esac

exit 0
