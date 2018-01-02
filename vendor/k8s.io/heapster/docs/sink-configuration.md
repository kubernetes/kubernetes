Configuring sinks
=================

Heapster can store data into different backends (sinks). These are specified on the command line
via the `--sink` flag. The flag takes an argument of the form `PREFIX:CONFIG[?OPTIONS]`.
Options (optional!) are specified as URL query parameters, separated by `&` as normal.
This allows each source to have custom configuration passed to it without needing to
continually add new flags to Heapster as new sinks are added. This also means
heapster can store data into multiple sinks at once.

## Current sinks

### Log

This sinks writes all data to the standard output which is particularly useful for debugging.

   --sink=log

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
* `withfields` - Use [InfluxDB fields](storage-schema.md#using-fields) (default: `false`)

### Google Cloud Monitoring
This sink supports monitoring metrics only.
To use the GCM sink add the following flag:

	--sink=gcm

*Note: This sink works only on a Google Compute Enginer VM as of now*

GCM has one option - `metrics` that can be set to:
* all - the sink exports all metrics
* autoscaling - the sink exports only autoscaling-related metrics

### Google Cloud Logging
This sink supports events only.
To use the InfluxDB sink add the following flag:

	--sink=gcl

*Notes:*
 * This sink works only on a Google Compute Enginer VM as of now
 * GCE instance must have “https://www.googleapis.com/auth/logging.write” auth scope
 * GCE instance must have Logging API enabled for the project in Google Developer Console
 * GCL Logs are accessible via:
    * `https://console.developers.google.com/project/<project_ID>/logs?service=custom.googleapis.com`
    * Where `project_ID` is the project ID of the Google Cloud Platform project ID.
    * Select `kubernetes.io/events` from the `All logs` drop down menu.

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
* `batchSize`- How many metrics are sent in each request to Hawkular-Metrics (default is 1000)
* `concurrencyLimit`- How many concurrent requests are used to send data to the Hawkular-Metrics (default is 5)

A combination of `insecure` / `caCert` / `auth` is not supported, only a single of these parameters is allowed at once. Also, combination of `useServiceAccount` and `user` + `pass` is not supported. To increase the performance of Hawkular sink in case of multiple instances of Hawkular-Metrics (such as scaled scenario in OpenShift) modify the parameters of batchSize and concurrencyLimit to balance the load on Hawkular-Metrics instances.

### OpenTSDB
This sink supports monitoring metrics and events.
To use the opentsdb sink add the following flag:

    --sink=opentsdb:<OPENTSDB_SERVER_URL>

Currently, accessing opentsdb via its rest apis doesn't need any authentication, so you
can enable opentsdb sink like this:

    --sink=opentsdb:http://192.168.1.8:4242

### Monasca
This sink supports monitoring metrics only.
To use the Monasca sink add the following flag:

	--sink=monasca:[?<OPTIONS>]

The available options are listed below, and some of them are mandatory. You need to provide access to the Identity service of OpenStack (keystone).
Currently, only authorization through `username` / `userID` + `password` / `APIKey` is supported.

The Monasca sink is then created with either the provided Monasca API Server URL, or the URL is discovered automatically if none is provided by the user.

The following options are available:

* `user-id` - ID of the OpenStack user
* `username` - Name of the OpenStack user
* `tenant-id` - ID of the OpenStack tenant (project)
* `keystone-url` - URL to the Keystone identity service (*mandatory*). Must be a v3 server (required by Monasca)
* `password` - Password of the OpenStack user
* `api-key` - API-Key for the OpenStack user
* `domain-id` - ID of the OpenStack user's domain
* `domain-name` - Name of the OpenStack user's domain
* `monasca-url` - URL of the Monasca API server (*optional*: the sink will attempt to discover the service if not provided)

### Kafka
This sink supports monitoring metrics only.
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
This sink supports metrics only.
To use the reimann sink add the following flag:

	--sink="riemann:<RIEMANN_SERVER_URL>[?<OPTIONS>]"

The following options are available:

* `ttl` - TTL for writes to Riemann. Default: `60 seconds`
* `state` - FIXME. Default: `""`
* `tags` - FIXME. Default. `none`
* `storeEvents` - Control storage of events. Default: `true`

### Elasticsearch
This sink supports monitoring metrics and events. To use the ElasticSearch
sink add the following flag:

    --sink=elasticsearch:<ES_SERVER_URL>[?<OPTIONS>]

Normally an ElasticSearch cluster has multiple nodes or a proxy, so these need
to be configured for the ElasticSearch sink. To do this, you can set
`ES_SERVER_URL` to a dummy value, and use the `?nodes=` query value for each
additional node in the cluster. For example:

  --sink=elasticsearch:?nodes=foo.com:9200&nodes=bar.com:9200

Besides this, the following options can be set in query string:

* `Index` - the index for metrics and events. The default is `heapster`
* `esUserName` - the username if authentication is enabled
* `esUserSecret` - the password if authentication is enabled
* `maxRetries` - the number of retries that the Elastic client will perform
  for a single request after before giving up and return an error. It is `0`
  by default, so retry is disabled by default.
* `healthCheck` - specifies if healthchecks are enabled by default. It is enabled
  by default. To disable, provide a negative boolean value like `0` or `false`.
* `sniff` - specifies if the sniffer is enabled by default. It is enabled
  by default. To disable, provide a negative boolean value like `0` or `false`.
* `startupHealthcheckTimeout` - the time in seconds the healthcheck waits for
  a response from Elasticsearch on startup, i.e. when creating a client. The
  default value is `1`.

Like this:

    --sink="elasticsearch:?nodes=0.0.0.0:9200&Index=testMetric"

	or

	--sink="elasticsearch:?nodes=0.0.0.0:9200&Index=testEvent"

## Using multiple sinks

Heapster can be configured to send k8s metrics and events to multiple sinks by specifying the`--sink=...` flag multiple times.

For example, to send data to both gcm and influxdb at the same time, you can use the following:

```shell
    --sink=gcm --sink=influxdb:http://monitoring-influxdb:80/
```
