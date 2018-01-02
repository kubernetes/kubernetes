# Monitoring etcd

Each etcd server exports metrics under the `/metrics` path on its client port.

The metrics can be fetched with `curl`:

```sh
$ curl -L http://localhost:2379/metrics

# HELP etcd_debugging_mvcc_keys_total Total number of keys.
# TYPE etcd_debugging_mvcc_keys_total gauge
etcd_debugging_mvcc_keys_total 0
# HELP etcd_debugging_mvcc_pending_events_total Total number of pending events to be sent.
# TYPE etcd_debugging_mvcc_pending_events_total gauge
etcd_debugging_mvcc_pending_events_total 0
...
```


## Prometheus

Running a [Prometheus][prometheus] monitoring service is the easiest way to ingest and record etcd's metrics.

First, install Prometheus:

```sh
PROMETHEUS_VERSION="1.3.1"
wget https://github.com/prometheus/prometheus/releases/download/v$PROMETHEUS_VERSION/prometheus-$PROMETHEUS_VERSION.linux-amd64.tar.gz -O /tmp/prometheus-$PROMETHEUS_VERSION.linux-amd64.tar.gz
tar -xvzf /tmp/prometheus-$PROMETHEUS_VERSION.linux-amd64.tar.gz --directory /tmp/ --strip-components=1
/tmp/prometheus -version
```

Set Prometheus's scraper to target the etcd cluster endpoints:

```sh
cat > /tmp/test-etcd.yaml <<EOF
global:
  scrape_interval: 10s
scrape_configs:
  - job_name: test-etcd
    static_configs:
    - targets: ['10.240.0.32:2379','10.240.0.33:2379','10.240.0.34:2379']
EOF
cat /tmp/test-etcd.yaml
```

Set up the Prometheus handler:

```sh
nohup /tmp/prometheus \
    -config.file /tmp/test-etcd.yaml \
    -web.listen-address ":9090" \
    -storage.local.path "test-etcd.data" >> /tmp/test-etcd.log  2>&1 &
```

Now Prometheus will scrape etcd metrics every 10 seconds.


## Grafana

[Grafana][grafana] has built-in Prometheus support; just add a Prometheus data source:

```
Name:   test-etcd
Type:   Prometheus
Url:    http://localhost:9090
Access: proxy
```

Then import the default [etcd dashboard template][template] and customize. For instance, if Prometheus data source name is `my-etcd`, the `datasource` field values in JSON also need to be `my-etcd`.

See the [demo][demo].

Sample dashboard:

![](./etcd-sample-grafana.png)


[prometheus]: https://prometheus.io/
[grafana]: http://grafana.org/
[template]: ./grafana.json
[demo]: http://dash.etcd.io/dashboard/db/test-etcd
