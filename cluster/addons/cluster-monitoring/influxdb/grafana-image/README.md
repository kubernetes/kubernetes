# Grafana Image For Kubernetes

## What's In It
* Stock Grafana.
* Logic to discover InfluxDB service endpoint and create a datasource for it.
* Create a couple of dashboards during startup. These dashboards leverage templating and repeating of panels features in Grafana 2.0, to discover nodes, pods, and containers automatically.

## How To Build It
### Build influxdb_service_discovery.go

```
cd $GOPATH/src/k8s.io/kubernetes/cluster/addons/cluster-monitoring/influxdb/grafana-image

docker run --rm -v "$GOPATH":/gopath -w /gopath/src/k8s.io/kubernetes/cluster/addons/cluster-monitoring/influxdb/grafana-image -e GOOS=linux -e GOARCH=amd64 -e GOPATH=/gopath golang:1.4 go build -v influxdb_service_discovery.go
```

This should create ```influxdb_service_discovery``` in the current folder.

### Build Docker Image

```
cd $GOPATH/src/k8s.io/kubernetes/cluster/addons/cluster-monitoring/influxdb/grafana-image

docker build -t gcr.io/google_containers/grafana:2.1.0 .

docker push gcr.io/google_containers/grafana:2.1.0
```

### Update The Spec
Make sure the YAML file refers to the latest version.

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/addons/cluster-monitoring/influxdb/grafana-image/README.md?pixel)]()
