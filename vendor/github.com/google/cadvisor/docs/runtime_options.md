# cAdvisor Runtime Options

This document describes a set of runtime flags available in cAdvisor.

## Local Storage Duration

cAdvisor stores the latest historical data in memory. How long of a history it stores can be configured with the `--storage_duration` flag.

```
--storage_duration: How long to store data.
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
```

## Container Hints

Container hints are a way to pass extra information about a container to cAdvisor. In this way cAdvisor can augment the stats it gathers. For more information on the container hints format see its [definition](container/raw/container_hints.go). Note that container hints are only used by the raw container driver today.

```
--container_hints="/etc/cadvisor/container_hints.json": location of the container hints file
```

## HTTP

Specify where cAdvisor listens.

```
--listen_ip="": IP to listen on, defaults to all IPs
--port=8080: port to listen
```

## Debugging and Logging

cAdvisor-native flags that help in debugging:

```
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

## Storage Drivers

See [InfluxDB instructions](influxdb.md).
