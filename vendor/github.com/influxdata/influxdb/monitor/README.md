# System Monitoring
_This functionality should be considered experimental and is subject to change._

_System Monitoring_ means all statistical and diagnostic information made availabe to the user of InfluxDB system, about the system itself. Its purpose is to assist with troubleshooting and performance analysis of the database itself.

## Statistics vs. Diagnostics
A distinction is made between _statistics_ and _diagnostics_ for the purposes of monitoring. Generally a statistical quality is something that is being counted, and for which it makes sense to store persistently for historical analysis. Diagnostic information is not necessarily numerical, and may not make sense to store.

An example of statistical information would be the number of points received over UDP, or the number of queries executed. Examples of diagnostic information would be a list of current Graphite TCP connections, the version of InfluxDB, or the uptime of the process.

## System Statistics
`SHOW STATS [FOR <module>]` displays statisics about subsystems within the running `influxd` process. Statistics include points received, points indexed, bytes written to disk, TCP connections handled etc. These statistics are all zero when the InfluxDB process starts. If _module_ is specified, it must be single-quoted. For example `SHOW STATS FOR 'httpd'`.

All statistics are written, by default, by each node to a "monitor" database within the InfluxDB system, allowing analysis of aggregated statistical data using the standard InfluxQL language. This allows users to track the performance of their system. Importantly, this allows cluster-level statistics to be viewed, since by querying the monitor database, statistics from all nodes may be queried. This can be a very powerful approach for troubleshooting your InfluxDB system and understanding its behaviour.

## System Diagnostics
`SHOW DIAGNOSTICS [FOR <module>]` displays various diagnostic information about the `influxd` process. This information is not stored persistently within the InfluxDB system. If _module_ is specified, it must be single-quoted. For example `SHOW STATS FOR 'build'`.

## Standard expvar support
All statistical information is available at HTTP API endpoint `/debug/vars`, in [expvar](https://golang.org/pkg/expvar/) format, allowing external systems to monitor an InfluxDB node. By default, the full path to this endpoint is `http://localhost:8086/debug/vars`.

## Configuration
The `monitor` module allows the following configuration:

 * Whether to write statistical and diagnostic information to an InfluxDB system. This is enabled by default.
 * The name of the database to where this information should be written. Defaults to `_internal`. The information is written to the default retention policy for the given database.
 * The name of the retention policy, along with full configuration control of the retention policy, if the default retention policy is not suitable.
 * The rate at which this information should be written. The default rate is once every 10 seconds.

# Design and Implementation

A new module named `monitor` supports all basic statistics and diagnostic functionality. This includes:

 * Allowing other modules to register statistics and diagnostics information, allowing it to be accessed on demand by the `monitor` module.
 * Serving the statistics and diagnostic information to the user, in response to commands such as `SHOW DIAGNOSTICS`.
 * Expose standard Go runtime information such as garbage collection statistics.
 * Make all collected expvar data via HTTP, for collection by 3rd-party tools.
 * Writing the statistical information to the "monitor" database, for query purposes.

## Registering statistics and diagnostics

To export statistical information with the `monitor` system, a service should implement the `monitor.Reporter` interface. Services added to the Server will be automatically added to the list of statistics returned. Any service that is not added to the `Services` slice will need to modify the `Server`'s `Statistics(map[string]string)` method to aggregate the call to the service's `Statistics(map[string]string)` method so they are combined into a single response. The `Statistics(map[string]string)` method should return a statistics slice with the passed in tags included. The statistics should be kept inside of an internal structure and should be accessed in a thread-safe way. It is common to create a struct for holding the statistics and using `sync/atomic` instead of locking. If using `sync/atomic`, be sure to align the values in the struct so it works properly on `i386`.

To register diagnostic information, `monitor.RegisterDiagnosticsClient` is called, passing a `influxdb.monitor.DiagsClient` object to `monitor`. Implementing the `influxdb.monitor.DiagsClient` interface requires that your component have function returning diagnostic information in specific form, so that it can be displayed by the `monitor` system.

Statistical information is reset to its initial state when a server is restarted.
