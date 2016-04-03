# Grafana Image For Heapster/InfluxDB

## What's In It
* Stock Grafana.
* Create a datasource for InfluxDB.
* Create a couple of dashboards during startup. These dashboards leverage templating and repeating of panels features in Grafana 2.0, to discover nodes, pods, and containers automatically.

## How To Use It
* InfluxDB service URL can be passed in via the environment variable __INFLUXDB_SERVICE_URL__.
* If __INFLUXDB_SERVICE_URL__ isn't defined, it will discover and use the external service URL, if available.
* Otherwise, it will fall back to http://monitoring-influxdb:8086.

## How To Build It

cd $GOPATH/src/k8s.io/heapster/grafana

make all
