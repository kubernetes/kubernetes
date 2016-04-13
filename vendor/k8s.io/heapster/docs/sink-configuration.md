Configuring sinks
===================

Heapster can store data into different backends (sinks). These are specified on the command line
via the `--sink` flag. The flag takes an argument of the form `PREFIX:CONFIG[?OPTIONS]`.
Options (optional!) are specified as URL query parameters, separated by `&` as normal.
This allows each source to have custom configuration passed to it without needing to
continually add new flags to Heapster as new sinks are added. This also means
heapster can store data into multiple sinks at once.

## Current sinks
### InfluxDB
This sink supports both monitoring metrics and events.
*Supports InfluxDB versions v0.9 and above*
To use the InfluxDB sink add the following flag:

	--sink=influxdb:<INFLUXDB_URL>[?<INFLUXDB_OPTIONS>]

If you're running Heapster in a Kubernetes cluster with the default InfluxDB + Grafana setup you can use the following flag:

	--sink=influxdb:http://monitoring-influxdb:80/

The following options are available:
* `user` - InfluxDB username (default: `root`)
* `pw` - InfluxDB password (default: `root`)
* `db` - InfluxDB Database name (default: `k8s`)
* `secure` - Connect securely to InfluxDB (default: `false`)

### Google Cloud Monitoring
This sink supports monitoring metrics only.
To use the GCM sink add the following flag:

	--sink=gcm

*Note: This sink works only on a Google Compute Enginer VM as of now*

This sink does not export any options!

### Google Cloud Monitoring Autoscaling
This sink supports monitoring metrics for autoscaling purposes only.
To use the GCM Autoscaling sink add the following flag:

	--sink=gcmautoscaling

*Note: This sink works only on a Google Compute Enginer VM as of now*

This sink does not export any options!

### Google Cloud Logging
This sink supports events only.
To use the InfluxDB sink add the following flag:

	--sink=gcl

*Note: This sink works only on a Google Compute Enginer VM as of now*

This sink does not export any options!

### Hawkular-Metrics
This sink supports monitoring metrics only.
To use the Hawkular-Metrics sink add the following flag:

	--sink=hawkular:<HAWKULAR_SERVER_URL>[?<OPTIONS>]

If `HAWKULAR_SERVER_URL` includes any path, the default `hawkular/metrics` is overridden. To use SSL, the `HAWKULAR_SERVER_URL` has to start with `https`

The following options are available:

* `tenant` - Hawkular-Metrics tenantId (default: `heapster`)
* `labelToTenant` - Hawkular-Metrics uses given label's value as tenant value when storing data
* `useServiceAccount` - Sink will use the service account token to authorize to Hawkular-Metrics (requires OpenShift)
* `insecure` - SSL connection will not verify the certificates
* `caCert` - A path to the CA Certificate file that will be used in the connection
* `auth` - Kubernetes authentication file that will be used for constructing the TLSConfig
* `user` - Username to connect to the Hawkular-Metrics server
* `pass` - Password to connect to the Hawkular-Metrics server
* `filter` - Allows bypassing the store of matching metrics, any number of `filter` parameters can be given with a syntax of `filter=operation(param)`. Supported operations and their params:
  * `label` - The syntax is `label(labelName:regexp)` where `labelName` is 1:1 match and `regexp` to use for matching is given after `:` delimiter
  * `name` - The syntax is `name(regexp)` where MetricName is matched (such as `cpu/usage`) with a `regexp` filter

A combination of `insecure` / `caCert` / `auth` is not supported, only a single of these parameters is allowed at once. Also, combination of `useServiceAccount` and `user` + `pass` is not supported.

### Kafka
This sink supports monitoring metrics and events.
To use the kafka sink add the following flag:

    --sink="kafka:<?<OPTIONS>>"

Normally, kafka server has multi brokers, so brokers' list need be configured for producer.
So, we provide kafka brokers' list and topics about timeseries & topic in url's query string.
Options can be set in query string, like this:

* `brokers` - Kafka's brokers' list. 
* `timeseriestopic` - Kafka's topic for timeseries. Default value : `heapster-metrics`
* `eventstopic` - Kafka's topic for events.Default value : `heapster-events`

For example, 

    --sink="kafka:?brokers=localhost:9092&brokers=localhost:9093&timeseriestopic=testseries&eventstopic=testtopic"

### Riemann
This sink supports metrics and events.
To use the reimann sink add the following flag:

	--sink="riemann:<RIEMANN_SERVER_URL>[?<OPTIONS>]"

The following options are available:

* `ttl` - TTL for writes to Riemann. Default: `60 seconds`
* `state` - FIXME. Default: `""`
* `tags` - FIXME. Default. `none`
* `storeEvents` - Control storage of events. Default: `true`

### OpenTSDB
This sink supports monitoring metrics and events.
To use the opentsdb sink add the following flag:

    --sink=opentsdb:<OPENTSDB_SERVER_URL>

Currently, accessing opentsdb via its rest apis doesn't need any authentication, so you
can enable opentsdb sink like this:

    --sink=opentsdb:http://192.168.1.8:4242

## Modifying the sinks at runtime

Using the `/api/v1/sinks` endpoint, it is possible to fetch the sinks
currently in use via a GET request or to change them via a POST request. The
format is the same as when passed via command line flags.

For example, to set gcm and influxdb as sinks, you may do the following:

```shell
echo '["gcm", "influxdb:http://monitoring-influxdb:8086"]' | curl \
    --insecure -u admin:<password> -X POST -d @- \
    -H "Accept: application/json" -H "Content-Type: application/json" \
    https://<master-ip>/api/v1/proxy/namespaces/kube-system/services/monitoring-heapster/api/v1/sinks
```

## Using multiple sinks

Heapster can be configured to send k8s metrics and events to multiple sinks by specifying the`--sink=...` flag multiple times.

For example, to send data to both gcm and influxdb at the same time, you can use the following:

```shell
    --sink=gcm --sink=influxdb:http://monitoring-influxdb:80/
```
