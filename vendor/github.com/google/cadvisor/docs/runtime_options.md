# cAdvisor Runtime Options

This document describes a set of runtime flags available in cAdvisor.

## Container Hints

Container hints are a way to pass extra information about a container to cAdvisor. In this way cAdvisor can augment the stats it gathers. For more information on the container hints format see its [definition](../container/common/container_hints.go). Note that container hints are only used by the raw container driver today.

```
--container_hints="/etc/cadvisor/container_hints.json": location of the container hints file
```

## CPU

```
--enable_load_reader=false: Whether to enable cpu load reader
--max_procs=0: max number of CPUs that can be used simultaneously. Less than 1 for default (number of cores).
```

## Debugging and Logging

cAdvisor-native flags that help in debugging:

```
--log_backtrace_at="": when logging hits line file:N, emit a stack trace
--log_cadvisor_usage=false: Whether to log the usage of the cAdvisor container
--version=false: print cAdvisor version and exit
--profiling=false: Enable profiling via web interface host:port/debug/pprof/
```

From [glog](https://github.com/golang/glog) here are some flags we find useful:

```
--log_dir="": If non-empty, write log files in this directory
--logtostderr=false: log to standard error instead of files
--alsologtostderr=false: log to standard error as well as files
--stderrthreshold=0: logs at or above this threshold go to stderr
--v=0: log level for V logs
--vmodule=: comma-separated list of pattern=N settings for file-filtered logging
```

## Docker

```
--docker="unix:///var/run/docker.sock": docker endpoint (default "unix:///var/run/docker.sock")
--docker_env_metadata_whitelist="": a comma-separated list of environment variable keys that needs to be collected for docker containers
--docker_only=false: Only report docker containers in addition to root stats
--docker_root="/var/lib/docker": DEPRECATED: docker root is read from docker info (this is a fallback, default: /var/lib/docker) (default "/var/lib/docker")
--docker-tls: use TLS to connect to docker
--docker-tls-cert="cert.pem": client certificate for TLS-connection with docker
--docker-tls-key="key.pem": private key for TLS-connection with docker
--docker-tls-ca="ca.pem": trusted CA for TLS-connection with docker
```

## Housekeeping

Housekeeping is the periodic actions cAdvisor takes. During these actions, cAdvisor will gather container stats. These flags control how and when cAdvisor performs housekeeping.

#### Dynamic Housekeeping

Dynamic housekeeping intervals let cAdvisor vary how often it gathers stats.
It does this depending on how active the container is. Turning this off
provides predictable housekeeping intervals, but increases the resource usage
of cAdvisor.

```
--allow_dynamic_housekeeping=true: Whether to allow the housekeeping interval to be dynamic
```

#### Housekeeping Intervals

Intervals for housekeeping. cAdvisor has two housekeepings: global and per-container.

Global housekeeping is a singular housekeeping done once in cAdvisor. This typically does detection of new containers. Today, cAdvisor discovers new containers with kernel events so this global housekeeping is mostly used as backup in the case that there are any missed events.

Per-container housekeeping is run once on each container cAdvisor tracks. This typically gets container stats.

```
--global_housekeeping_interval=1m0s: Interval between global housekeepings
--housekeeping_interval=1s: Interval between container housekeepings
--max_housekeeping_interval=1m0s: Largest interval to allow between container housekeepings (default 1m0s)
```

## HTTP

Specify where cAdvisor listens.

```
--http_auth_file="": HTTP auth file for the web UI
--http_auth_realm="localhost": HTTP auth realm for the web UI (default "localhost")
--http_digest_file="": HTTP digest file for the web UI
--http_digest_realm="localhost": HTTP digest file for the web UI (default "localhost")
--listen_ip="": IP to listen on, defaults to all IPs
--port=8080: port to listen (default 8080)
```

## Local Storage Duration

cAdvisor stores the latest historical data in memory. How long of a history it stores can be configured with the `--storage_duration` flag.

```
--storage_duration=2m0s: How long to store data.
```

## Machine

```
--boot_id_file="/proc/sys/kernel/random/boot_id": Comma-separated list of files to check for boot-id. Use the first one that exists. (default "/proc/sys/kernel/random/boot_id")
--machine_id_file="/etc/machine-id,/var/lib/dbus/machine-id": Comma-separated list of files to check for machine-id. Use the first one that exists. (default "/etc/machine-id,/var/lib/dbus/machine-id")
```

## Metrics

```
--application_metrics_count_limit=100: Max number of application metrics to store (per container) (default 100)
--collector_cert="": Collector's certificate, exposed to endpoints for certificate based authentication.
--collector_key="": Key for the collector's certificate
--disable_metrics=tcp, udp: comma-separated list of metrics to be disabled. Options are 'disk', 'network', 'tcp', 'udp'. Note: tcp and udp are disabled by default due to high CPU usage. (default tcp,udp)
--prometheus_endpoint="/metrics": Endpoint to expose Prometheus metrics on (default "/metrics")
```

## Storage Drivers

```
--storage_driver="": Storage driver to use. Data is always cached shortly in memory, this controls where data is pushed besides the local cache. Empty means none. Options are: <empty>, bigquery, elasticsearch, influxdb, kafka, redis, statsd, stdout
--storage_driver_buffer_duration="1m0s": Writes in the storage driver will be buffered for this duration, and committed to the non memory backends as a single transaction (default 1m0s)
--storage_driver_db="cadvisor": database name (default "cadvisor")
--storage_driver_host="localhost:8086": database host:port (default "localhost:8086")
--storage_driver_password="root": database password (default "root")
--storage_driver_secure=false: use secure connection with database
--storage_driver_table="stats": table name (default "stats")
--storage_driver_user="root": database username (default "root")
```

For storage driver specific instructions:

* [InfluxDB instructions](storage/influxdb.md).
* [ElasticSearch instructions](storage/elasticsearch.md).
* [Kafka instructions](storage/kafka.md).
* [Prometheus instructions](storage/prometheus.md).
