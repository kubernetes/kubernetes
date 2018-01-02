## v1.1.1 [2016-12-06]

### Features

- [#7684](https://github.com/influxdata/influxdb/issues/7684): Update Go version to 1.7.4.

### Bugfixes

- [#7625](https://github.com/influxdata/influxdb/issues/7625): Fix incorrect tag value in error message.
- [#7661](https://github.com/influxdata/influxdb/pull/7661): Quote the empty string as an ident.
- [#7679](https://github.com/influxdata/influxdb/pull/7679): Fix string fields w/ trailing slashes

### Security

[Go 1.7.4](https://golang.org/doc/devel/release.html#go1.7.minor) was released to address two security issues.  This release includes these security fixes.

## v1.1.0 [2016-11-14]

### Release Notes

This release is built with go 1.7.3 and provides many performance optimizations, stability changes and a few new query capabilities.  If upgrading from a prior version, please read the configuration changes below section before upgrading.

### Deprecations

The admin interface is deprecated and will be removed in a subsequent release.  The configuration setting to enable the admin UI is now disabled by default, but can be enabled if necessary.  We recommend using [Chronograf](https://github.com/influxdata/chronograf) or [Grafana](https://github.com/grafana/grafana) as a replacement.

### Configuration Changes

The following configuration changes may need to changed before upgrading to `1.1.0` from prior versions.

#### `[admin]` Section

* `enabled` now default to false.  If you are currently using the admin interaface, you will need to change this value to `true` to re-enable it.  The admin interface is currently deprecated and will be removed in a subsequent release.

#### `[data]` Section

* `max-values-per-tag` was added with a default of 100,000, but can be disabled by setting it to `0`.  Existing measurements with tags that exceed this limit will continue to load, but writes that would cause the tags cardinality to increase will be dropped and a `partial write` error will be returned to the caller.  This limit can be used to prevent high cardinality tag values from being written to a measurement.
* `cache-max-memory-size` has been increased to from `524288000` to `1048576000`.  This setting is the maximum amount of RAM, in bytes, a shard cache can use before it rejects writes with an error.  Setting this value to `0` disables the limit.
* `cache-snapshot-write-cold-duration` has been decreased from `1h` to `10m`.  This setting determines how long values will stay in the shard cache while the shard is cold for writes.
* `compact-full-write-cold-duration` has been decreased from `24h` to `4h`.  The shorter duration allows cold shards to be compacted to an optimal state more quickly.

### Features

The query language has been extended with a few new features:

* New `cumulative_sum` function - [#7388](https://github.com/influxdata/influxdb/pull/7388)
* New `linear` fill option - [#7408](https://github.com/influxdata/influxdb/pull/7408)
* Support `ON` for `SHOW` commands - [#7295](https://github.com/influxdata/influxdb/pull/7295)
* Support regex on fields keys in select clause - [#7442](https://github.com/influxdata/influxdb/pull/7442)

All Changes:

- [#7415](https://github.com/influxdata/influxdb/pull/7415): Add sample function to query language.
- [#7403](https://github.com/influxdata/influxdb/pull/7403): Add `fill(linear)` to query language.
- [#7120](https://github.com/influxdata/influxdb/issues/7120): Add additional statistics to query executor.
- [#7135](https://github.com/influxdata/influxdb/pull/7135): Support enable HTTP service over unix domain socket. Thanks @oiooj
- [#3634](https://github.com/influxdata/influxdb/issues/3634): Support mixed duration units.
- [#7099](https://github.com/influxdata/influxdb/pull/7099): Implement text/csv content encoding for the response writer.
- [#6992](https://github.com/influxdata/influxdb/issues/6992): Support tools for running async queries.
- [#7136](https://github.com/influxdata/influxdb/pull/7136): Update jwt-go dependency to version 3.
- [#6962](https://github.com/influxdata/influxdb/issues/6962): Support ON and use default database for SHOW commands.
- [#7268](https://github.com/influxdata/influxdb/pull/7268): More man pages for the other tools we package and compress man pages fully.
- [#7305](https://github.com/influxdata/influxdb/pull/7305): UDP Client: Split large points. Thanks @vlasad
- [#7115](https://github.com/influxdata/influxdb/issues/7115): Feature request: `influx inspect -export` should dump WAL files.
- [#7388](https://github.com/influxdata/influxdb/pull/7388): Implement cumulative_sum() function.
- [#7441](https://github.com/influxdata/influxdb/pull/7441): Speed up shutdown by closing shards concurrently.
- [#7146](https://github.com/influxdata/influxdb/issues/7146): Add max-values-per-tag to limit high tag cardinality data
- [#5955](https://github.com/influxdata/influxdb/issues/5955): Make regex work on field and dimension keys in SELECT clause.
- [#7470](https://github.com/influxdata/influxdb/pull/7470): Reduce map allocations when computing the TagSet of a measurement.
- [#6894](https://github.com/influxdata/influxdb/issues/6894): Support `INFLUX_USERNAME` and `INFLUX_PASSWORD` for setting username/password in the CLI.
- [#6896](https://github.com/influxdata/influxdb/issues/6896): Correctly read in input from a non-interactive stream for the CLI.
- [#7463](https://github.com/influxdata/influxdb/pull/7463): Make input plugin services open/close idempotent.
- [#7473](https://github.com/influxdata/influxdb/pull/7473): Align binary math expression streams by time.
- [#7281](https://github.com/influxdata/influxdb/pull/7281): Add stats for active compactions, compaction errors.
- [#7496](https://github.com/influxdata/influxdb/pull/7496): Filter out series within shards that do not have data for that series.
- [#7480](https://github.com/influxdata/influxdb/pull/7480): Improve compaction planning performance by caching tsm file stats.
- [#7320](https://github.com/influxdata/influxdb/issues/7320): Update defaults in config for latest best practices
- [#7495](https://github.com/influxdata/influxdb/pull/7495): Rewrite regexes of the form host = /^server-a$/ to host = 'server-a', to take advantage of the tsdb index.
- [#6704](https://github.com/influxdata/influxdb/issues/6704): Optimize first/last when no group by interval is present.
- [#4461](https://github.com/influxdata/influxdb/issues/4461): Change default time boundaries for raw queries.

### Bugfixes

- [#7392](https://github.com/influxdata/influxdb/pull/7392): Enable https subscriptions to work with custom CA certificates.
- [#1834](https://github.com/influxdata/influxdb/issues/1834): Drop time when used as a tag or field key.
- [#7152](https://github.com/influxdata/influxdb/issues/7152): Decrement number of measurements only once when deleting the last series from a measurement.
- [#7177](https://github.com/influxdata/influxdb/issues/7177): Fix base64 encoding issue with /debug/vars stats.
- [#7196](https://github.com/influxdata/influxdb/issues/7196): Fix mmap dereferencing, fixes #7183, #7180
- [#7013](https://github.com/influxdata/influxdb/issues/7013): Fix the dollar sign so it properly handles reserved keywords.
- [#7297](https://github.com/influxdata/influxdb/issues/7297): Use consistent column output from the CLI for column formatted responses.
- [#7231](https://github.com/influxdata/influxdb/issues/7231): Duplicate parsing bug in ALTER RETENTION POLICY.
- [#7285](https://github.com/influxdata/influxdb/issues/7285): Correctly use password-type field in Admin UI. Thanks @dandv!
- [#2792](https://github.com/influxdata/influxdb/issues/2792): Exceeding max retention policy duration gives incorrect error message
- [#7226](https://github.com/influxdata/influxdb/issues/7226): Fix database locked up when deleting shards
- [#7382](https://github.com/influxdata/influxdb/issues/7382): Shard stats include wal path tag so disk bytes make more sense.
- [#7385](https://github.com/influxdata/influxdb/pull/7385): Reduce query planning allocations
- [#7436](https://github.com/influxdata/influxdb/issues/7436): Remove accidentally added string support for the stddev call.
- [#7161](https://github.com/influxdata/influxdb/issues/7161): Drop measurement causes cache max memory exceeded error.
- [#7334](https://github.com/influxdata/influxdb/issues/7334): Panic with unread show series iterators during drop database
- [#7482](https://github.com/influxdata/influxdb/issues/7482): Fix issue where point would be written to wrong shard.
- [#7431](https://github.com/influxdata/influxdb/issues/7431): Remove /data/process_continuous_queries endpoint.
- [#7053](https://github.com/influxdata/influxdb/issues/7053): Delete statement returns an error when retention policy or database is specified
- [#7494](https://github.com/influxdata/influxdb/issues/7494): influx_inspect: export does not escape field keys.
- [#7526](https://github.com/influxdata/influxdb/issues/7526): Truncate the version string when linking to the documentation.
- [#7548](https://github.com/influxdata/influxdb/issues/7548): Fix output duration units for SHOW QUERIES.
- [#7564](https://github.com/influxdata/influxdb/issues/7564): Fix incorrect grouping when multiple aggregates are used with sparse data.
- [#7606](https://github.com/influxdata/influxdb/pull/7606): Avoid deadlock when `max-row-limit` is hit.

## v1.0.2 [2016-10-05]

### Bugfixes

- [#7150](https://github.com/influxdata/influxdb/issues/7150): Do not automatically reset the shard duration when using ALTER RETENTION POLICY
- [#5878](https://github.com/influxdata/influxdb/issues/5878): Ensure correct shard groups created when retention policy has been altered.
- [#7391](https://github.com/influxdata/influxdb/issues/7391): Fix RLE integer decoding producing negative numbers
- [#7335](https://github.com/influxdata/influxdb/pull/7335): Avoid stat syscall when planning compactions
- [#7330](https://github.com/influxdata/influxdb/issues/7330): Subscription data loss under high write load

## v1.0.1 [2016-09-26]

### Bugfixes

- [#7271](https://github.com/influxdata/influxdb/issues/7271): Fixing typo within example configuration file. Thanks @andyfeller!
- [#7270](https://github.com/influxdata/influxdb/issues/7270): Implement time math for lazy time literals.
- [#7272](https://github.com/influxdata/influxdb/issues/7272): Report cmdline and memstats in /debug/vars.
- [#7299](https://github.com/influxdata/influxdb/ssues/7299): Ensure fieldsCreated stat available in shard measurement.
- [#6846](https://github.com/influxdata/influxdb/issues/6846): Read an invalid JSON response as an error in the influx client.
- [#7110](https://github.com/influxdata/influxdb/issues/7110): Skip past points at the same time in derivative call within a merged series.
- [#7226](https://github.com/influxdata/influxdb/issues/7226): Fix database locked up when deleting shards
- [#7315](https://github.com/influxdata/influxdb/issues/7315): Prevent users from manually using system queries since incorrect use would result in a panic.

## v1.0.0 [2016-09-08]

### Release Notes

### Breaking changes

* `max-series-per-database` was added with a default of 1M but can be disabled by setting it to `0`. Existing databases with series that exceed this limit will continue to load but writes that would create new series will fail.
* Config option `[cluster]` has been replaced with `[coordinator]`
* Support for config options `[collectd]` and `[opentsdb]` has been removed; use `[[collectd]]` and `[[opentsdb]]` instead.
* Config option `data-logging-enabled` within the `[data]` section, has been renamed to `trace-logging-enabled`, and defaults to `false`.
* The keywords `IF`, `EXISTS`, and `NOT` where removed for this release.  This means you no longer need to specify `IF NOT EXISTS` for `DROP DATABASE` or `IF EXISTS` for `CREATE DATABASE`.  If these are specified, a query parse error is returned.
* The Shard `writePointsFail` stat has been renamed to `writePointsErr` for consistency with other stats.

With this release the systemd configuration files for InfluxDB will use the system configured default for logging and will no longer write files to `/var/log/influxdb` by default. On most systems, the logs will be directed to the systemd journal and can be accessed by `journalctl -u influxdb.service`. Consult the systemd journald documentation for configuring journald.

### Features

- [#3541](https://github.com/influxdata/influxdb/issues/3451): Update SHOW FIELD KEYS to return the field type with the field key.
- [#6609](https://github.com/influxdata/influxdb/pull/6609): Add support for JWT token authentication.
- [#6559](https://github.com/influxdata/influxdb/issues/6559): Teach the http service how to enforce connection limits.
- [#6623](https://github.com/influxdata/influxdb/pull/6623): Speed up drop database
- [#6519](https://github.com/influxdata/influxdb/issues/6519): Support cast syntax for selecting a specific type.
- [#6654](https://github.com/influxdata/influxdb/pull/6654): Add new HTTP statistics to monitoring
- [#6664](https://github.com/influxdata/influxdb/pull/6664): Adds monitoring statistic for on-disk shard size.
- [#2926](https://github.com/influxdata/influxdb/issues/2926): Support bound parameters in the parser.
- [#1310](https://github.com/influxdata/influxdb/issues/1310): Add https-private-key option to httpd config.
- [#6621](https://github.com/influxdata/influxdb/pull/6621): Add Holt-Winter forecasting function.
- [#6655](https://github.com/influxdata/influxdb/issues/6655): Add HTTP(s) based subscriptions.
- [#5906](https://github.com/influxdata/influxdb/issues/5906): Dynamically update the documentation link in the admin UI.
- [#6686](https://github.com/influxdata/influxdb/pull/6686): Optimize timestamp run-length decoding
- [#6713](https://github.com/influxdata/influxdb/pull/6713): Reduce allocations during query parsing.
- [#3733](https://github.com/influxdata/influxdb/issues/3733): Modify the default retention policy name and make it configurable.
- [#6812](https://github.com/influxdata/influxdb/pull/6812): Make httpd logger closer to Common (& combined) Log Format.
- [#5655](https://github.com/influxdata/influxdb/issues/5655): Support specifying a retention policy for the graphite service.
- [#6820](https://github.com/influxdata/influxdb/issues/6820): Add NodeID to execution options
- [#4532](https://github.com/influxdata/influxdb/issues/4532): Support regex selection in SHOW TAG VALUES for the key.
- [#6889](https://github.com/influxdata/influxdb/pull/6889): Update help and remove unused config options from the configuration file.
- [#6900](https://github.com/influxdata/influxdb/pull/6900): Trim BOM from Windows Notepad-saved config files.
- [#6938](https://github.com/influxdata/influxdb/issues/6938): Added favicon
- [#6507](https://github.com/influxdata/influxdb/issues/6507): Refactor monitor service to avoid expvar and write monitor statistics on a truncated time interval.
- [#6805](https://github.com/influxdata/influxdb/issues/6805): Allow any variant of the help option to trigger the help.
- [#5499](https://github.com/influxdata/influxdb/issues/5499): Add stats and diagnostics to the TSM engine.
- [#6959](https://github.com/influxdata/influxdb/issues/6959): Return 403 Forbidden when authentication succeeds but authorization fails.
- [#1110](https://github.com/influxdata/influxdb/issues/1110): Support loading a folder for collectd typesdb files.
- [#6928](https://github.com/influxdata/influxdb/issues/6928): Run continuous query for multiple buckets rather than one per bucket.
- [#5500](https://github.com/influxdata/influxdb/issues/5500): Add extra trace logging to tsm engine.
- [#6909](https://github.com/influxdata/influxdb/issues/6909): Log the CQ execution time when continuous query logging is enabled.
- [#7046](https://github.com/influxdata/influxdb/pull/7046): Add tsm file export to influx_inspect tool.
- [#7011](https://github.com/influxdata/influxdb/issues/7011): Create man pages for commands.
- [#7050](https://github.com/influxdata/influxdb/pull/7050): Update go package library dependencies.
- [#5750](https://github.com/influxdata/influxdb/issues/5750): Support wildcards in aggregate functions.
- [#7065](https://github.com/influxdata/influxdb/issues/7065): Remove IF EXISTS/IF NOT EXISTS from influxql language.
- [#7095](https://github.com/influxdata/influxdb/pull/7095): Add MaxSeriesPerDatabase config setting.
- [#7199](https://github.com/influxdata/influxdb/pull/7199): Add mode function. Thanks @agaurav.
- [#7194](https://github.com/influxdata/influxdb/issues/7194): Support negative timestamps for the query engine.
- [#7172](https://github.com/influxdata/influxdb/pull/7172): Write path stats

### Bugfixes

- [#6604](https://github.com/influxdata/influxdb/pull/6604): Remove old cluster code
- [#6618](https://github.com/influxdata/influxdb/pull/6618): Optimize shard loading
- [#6629](https://github.com/influxdata/influxdb/issues/6629): query-log-enabled in config not ignored anymore.
- [#6607](https://github.com/influxdata/influxdb/issues/6607): SHOW TAG VALUES accepts != and !~ in WHERE clause.
- [#6649](https://github.com/influxdata/influxdb/issues/6649): Make sure admin exists before authenticating query.
- [#6644](https://github.com/influxdata/influxdb/issues/6644): Print the query executor's stack trace on a panic to the log.
- [#6650](https://github.com/influxdata/influxdb/issues/6650): Data race when dropping a database immediately after writing to it
- [#6235](https://github.com/influxdata/influxdb/issues/6235): Fix measurement field panic in tsm1 engine.
- [#6663](https://github.com/influxdata/influxdb/issues/6663): Fixing panic in SHOW FIELD KEYS.
- [#6624](https://github.com/influxdata/influxdb/issues/6624): Ensure clients requesting gzip encoded bodies don't receive empty body
- [#6652](https://github.com/influxdata/influxdb/issues/6652): Fix panic: interface conversion: tsm1.Value is \*tsm1.StringValue, not \*tsm1.FloatValue
- [#6406](https://github.com/influxdata/influxdb/issues/6406): Max index entries exceeded
- [#6557](https://github.com/influxdata/influxdb/issues/6557): Overwriting points on large series can cause memory spikes during compactions
- [#6611](https://github.com/influxdata/influxdb/issues/6611): Queries slow down hundreds times after overwriting points
- [#6641](https://github.com/influxdata/influxdb/issues/6641): Fix read tombstones: EOF
- [#6661](https://github.com/influxdata/influxdb/issues/6661): Disable limit optimization when using an aggregate.
- [#6676](https://github.com/influxdata/influxdb/issues/6676): Ensures client sends correct precision when inserting points.
- [#2048](https://github.com/influxdata/influxdb/issues/2048): Check that retention policies exist before creating CQ
- [#6702](https://github.com/influxdata/influxdb/issues/6702): Fix SELECT statement required privileges.
- [#6701](https://github.com/influxdata/influxdb/issues/6701): Filter out sources that do not match the shard database/retention policy.
- [#6683](https://github.com/influxdata/influxdb/issues/6683): Fix compaction planning re-compacting large TSM files
- [#6693](https://github.com/influxdata/influxdb/pull/6693): Truncate the shard group end time if it exceeds MaxNanoTime.
- [#6672](https://github.com/influxdata/influxdb/issues/6672): Accept points with trailing whitespace.
- [#6599](https://github.com/influxdata/influxdb/issues/6599): Ensure that future points considered in SHOW queries.
- [#6720](https://github.com/influxdata/influxdb/issues/6720): Concurrent map read write panic. Thanks @arussellsaw
- [#6727](https://github.com/influxdata/influxdb/issues/6727): queries with strings that look like dates end up with date types, not string types
- [#6250](https://github.com/influxdata/influxdb/issues/6250): Slow startup time
- [#6753](https://github.com/influxdata/influxdb/issues/6753): Prevent panic if there are no values.
- [#6685](https://github.com/influxdata/influxdb/issues/6685): Batch SELECT INTO / CQ writes
- [#6756](https://github.com/influxdata/influxdb/issues/6756): Set X-Influxdb-Version header on every request (even 404 requests).
- [#6760](https://github.com/influxdata/influxdb/issues/6760): Prevent panic in concurrent auth cache write
- [#6771](https://github.com/influxdata/influxdb/issues/6771): Fix the point validation parser to identify and sort tags correctly.
- [#6835](https://github.com/influxdata/influxdb/pull/6835): Include sysvinit-tools as an rpm dependency.
- [#6834](https://github.com/influxdata/influxdb/pull/6834): Add port to all graphite log output to help with debugging multiple endpoints
- [#6850](https://github.com/influxdata/influxdb/pull/6850): Modify the max nanosecond time to be one nanosecond less.
- [#6824](https://github.com/influxdata/influxdb/issues/6824): Remove systemd output redirection.
- [#6859](https://github.com/influxdata/influxdb/issues/6859): Set the condition cursor instead of aux iterator when creating a nil condition cursor.
- [#6869](https://github.com/influxdata/influxdb/issues/6869): Remove FieldCodec from tsdb package.
- [#6882](https://github.com/influxdata/influxdb/pull/6882): Remove a double lock in the tsm1 index writer.
- [#6883](https://github.com/influxdata/influxdb/pull/6883): Rename dumptsmdev to dumptsm in influx_inspect.
- [#6864](https://github.com/influxdata/influxdb/pull/6864): Allow a non-admin to call "use" for the influx cli.
- [#6855](https://github.com/influxdata/influxdb/pull/6855): Update `stress/v2` to work with clusters, ssl, and username/password auth. Code cleanup
- [#6738](https://github.com/influxdata/influxdb/issues/6738): Time sorting broken with overwritten points
- [#6829](https://github.com/influxdata/influxdb/issues/6829): Fix panic: runtime error: index out of range
- [#6911](https://github.com/influxdata/influxdb/issues/6911): Fix fill(previous) when used with math operators.
- [#6934](https://github.com/influxdata/influxdb/pull/6934): Fix regex binary encoding for a measurement.
- [#6942](https://github.com/influxdata/influxdb/pull/6942): Fix panic: truncate the slice when merging the caches.
- [#6708](https://github.com/influxdata/influxdb/issues/6708): Drop writes from before the retention policy time window.
- [#6968](https://github.com/influxdata/influxdb/issues/6968): Always use the demo config when outputting a new config.
- [#6986](https://github.com/influxdata/influxdb/pull/6986): update connection settings when changing hosts in cli.
- [#6965](https://github.com/influxdata/influxdb/pull/6965): Minor improvements to init script. Removes sysvinit-utils as package dependency.
- [#6952](https://github.com/influxdata/influxdb/pull/6952): Fix compaction planning with large TSM files
- [#6819](https://github.com/influxdata/influxdb/issues/6819): Database unresponsive after DROP MEASUREMENT
- [#6796](https://github.com/influxdata/influxdb/issues/6796): Out of Memory Error when Dropping Measurement
- [#6946](https://github.com/influxdata/influxdb/issues/6946): Duplicate data for the same timestamp
- [#7043](https://github.com/influxdata/influxdb/pull/7043): Remove limiter from walkShards
- [#5501](https://github.com/influxdata/influxdb/issues/5501): Queries against files that have just been compacted need to point to new files
- [#6595](https://github.com/influxdata/influxdb/issues/6595): Fix full compactions conflicting with level compactions
- [#7081](https://github.com/influxdata/influxdb/issues/7081): Hardcode auto generated RP names to autogen
- [#7088](https://github.com/influxdata/influxdb/pull/7088): Fix UDP pointsRx being incremented twice.
- [#7080](https://github.com/influxdata/influxdb/pull/7080): Ensure IDs can't clash when managing Continuous Queries.
- [#6990](https://github.com/influxdata/influxdb/issues/6990): Fix panic parsing empty key
- [#7084](https://github.com/influxdata/influxdb/pull/7084): Tombstone memory improvements
- [#6543](https://github.com/influxdata/influxdb/issues/6543): Fix parseFill to check for fill ident before attempting to parse an expression.
- [#7032](https://github.com/influxdata/influxdb/pull/7032): Copy tags in influx_stress to avoid a concurrent write panic on a map.
- [#7028](https://github.com/influxdata/influxdb/pull/7028): Do not run continuous queries that have no time span.
- [#7025](https://github.com/influxdata/influxdb/issues/7025): Move the CQ interval by the group by offset.
- [#7125](https://github.com/influxdata/influxdb/pull/7125): Ensure gzip writer is closed in influx_inspect export
- [#7127](https://github.com/influxdata/influxdb/pull/7127): Concurrent series limit
- [#7119](https://github.com/influxdata/influxdb/pull/7119): Fix CREATE DATABASE when dealing with default values.
- [#7218](https://github.com/influxdata/influxdb/issues/7218): Fix alter retention policy when all options are used.
- [#7225](https://github.com/influxdata/influxdb/issues/7225): runtime: goroutine stack exceeds 1000000000-byte limit
- [#7240](https://github.com/influxdata/influxdb/issues/7240): Allow blank lines in the line protocol input.
- [#7119](https://github.com/influxdata/influxdb/pull/7119): Fix CREATE DATABASE when dealing with default values.
- [#7243](https://github.com/influxdata/influxdb/issues/7243): Optimize queries that compare a tag value to an empty string.
- [#7074](https://github.com/influxdata/influxdb/issues/7074): Continuous full compactions

## v0.13.0 [2016-05-12]

### Release Notes

With this release InfluxDB is moving to Go v1.6.

### Features

- [#6213](https://github.com/influxdata/influxdb/pull/6213): Make logging output location more programmatically configurable.
- [#6237](https://github.com/influxdata/influxdb/issues/6237): Enable continuous integration testing on Windows platform via AppVeyor. Thanks @mvadu
- [#6263](https://github.com/influxdata/influxdb/pull/6263): Reduce UDP Service allocation size.
- [#6228](https://github.com/influxdata/influxdb/pull/6228): Support for multiple listeners for collectd and OpenTSDB inputs.
- [#6292](https://github.com/influxdata/influxdb/issues/6292): Allow percentile to be used as a selector.
- [#5707](https://github.com/influxdata/influxdb/issues/5707): Return a deprecated message when IF NOT EXISTS is used.
- [#6334](https://github.com/influxdata/influxdb/pull/6334): Allow environment variables to be set per input type.
- [#6394](https://github.com/influxdata/influxdb/pull/6394): Allow time math with integer timestamps.
- [#3247](https://github.com/influxdata/influxdb/issues/3247): Implement derivatives across intervals for aggregate queries.
- [#3166](https://github.com/influxdata/influxdb/issues/3166): Sort the series keys inside of a tag set so output is deterministic.
- [#1856](https://github.com/influxdata/influxdb/issues/1856): Add `elapsed` function that returns the time delta between subsequent points.
- [#5502](https://github.com/influxdata/influxdb/issues/5502): Add checksum verification to TSM inspect tool
- [#6444](https://github.com/influxdata/influxdb/pull/6444): Allow setting the config path through an environment variable and default config path.
- [#3558](https://github.com/influxdata/influxdb/issues/3558): Support field math inside a WHERE clause.
- [#6429](https://github.com/influxdata/influxdb/issues/6429): Log slow queries if they pass a configurable threshold.
- [#4675](https://github.com/influxdata/influxdb/issues/4675): Allow derivative() function to be used with ORDER BY desc.
- [#6483](https://github.com/influxdata/influxdb/pull/6483): Delete series support for TSM
- [#6484](https://github.com/influxdata/influxdb/pull/6484): Query language support for DELETE
- [#6290](https://github.com/influxdata/influxdb/issues/6290): Add POST /query endpoint and warning messages for using GET with write operations.
- [#6494](https://github.com/influxdata/influxdb/issues/6494): Support booleans for min() and max().
- [#2074](https://github.com/influxdata/influxdb/issues/2074): Support offset argument in the GROUP BY time(...) call.
- [#6533](https://github.com/influxdata/influxdb/issues/6533): Optimize SHOW SERIES
- [#6534](https://github.com/influxdata/influxdb/pull/6534): Move to Go v1.6.2 (over Go v1.4.3)
- [#6522](https://github.com/influxdata/influxdb/pull/6522): Dump TSM files to line protocol
- [#6585](https://github.com/influxdata/influxdb/pull/6585): Parallelize iterators
- [#6502](https://github.com/influxdata/influxdb/pull/6502): Add ability to copy shard via rpc calls.  Remove deprecated copier service.
- [#6593](https://github.com/influxdata/influxdb/pull/6593): Add ability to create snapshots of shards.

### Bugfixes

- [#6283](https://github.com/influxdata/influxdb/pull/6283): Fix GROUP BY tag to produce consistent results when a series has no tags.
- [#3773](https://github.com/influxdata/influxdb/issues/3773): Support empty tags for all WHERE equality operations.
- [#6270](https://github.com/influxdata/influxdb/issues/6270): tsm1 query engine alloc reduction
- [#6287](https://github.com/influxdata/influxdb/issues/6287): Fix data race in Influx Client.
- [#6252](https://github.com/influxdata/influxdb/pull/6252): Remove TSDB listener accept message @simnv
- [#6202](https://github.com/influxdata/influxdb/pull/6202): Check default SHARD DURATION when recreating the same database.
- [#6296](https://github.com/influxdata/influxdb/issues/6296): Allow the implicit time field to be renamed again.
- [#6294](https://github.com/influxdata/influxdb/issues/6294): Fix panic running influx_inspect info.
- [#6382](https://github.com/influxdata/influxdb/pull/6382): Removed dead code from the old query engine.
- [#3369](https://github.com/influxdata/influxdb/issues/3369): Detect when a timer literal will overflow or underflow the query engine.
- [#6398](https://github.com/influxdata/influxdb/issues/6398): Fix CREATE RETENTION POLICY parsing so it doesn't consume tokens it shouldn't.
- [#6425](https://github.com/influxdata/influxdb/pull/6425): Close idle tcp connections in HTTP client to prevent tcp conn leak.
- [#6109](https://github.com/influxdata/influxdb/issues/6109): Cache maximum memory size exceeded on startup
- [#6427](https://github.com/influxdata/influxdb/pull/6427): Fix setting uint config options via env vars
- [#6458](https://github.com/influxdata/influxdb/pull/6458): Make it clear when the CLI version is unknown.
- [#3883](https://github.com/influxdata/influxdb/issues/3883): Improve query sanitization to prevent a password leak in the logs.
- [#6462](https://github.com/influxdata/influxdb/pull/6462): Add safer locking to CreateFieldIfNotExists
- [#6361](https://github.com/influxdata/influxdb/pull/6361): Fix cluster/pool release of connection
- [#6470](https://github.com/influxdata/influxdb/pull/6470): Remove SHOW SERVERS & DROP SERVER support
- [#6477](https://github.com/influxdata/influxdb/pull/6477): Don't catch SIGQUIT or SIGHUP signals.
- [#6468](https://github.com/influxdata/influxdb/issues/6468): Panic with truncated wal segments
- [#6491](https://github.com/influxdata/influxdb/pull/6491): Fix the CLI not to enter an infinite loop when the liner has an error.
- [#6457](https://github.com/influxdata/influxdb/issues/6457): Retention policy cleanup does not remove series
- [#6477](https://github.com/influxdata/influxdb/pull/6477): Don't catch SIGQUIT or SIGHUP signals.
- [#6468](https://github.com/influxdata/influxdb/issues/6468): Panic with truncated wal segments
- [#6480](https://github.com/influxdata/influxdb/issues/6480): Fix SHOW statements' rewriting bug
- [#6505](https://github.com/influxdata/influxdb/issues/6505): Add regex literal to InfluxQL spec for FROM clause.
- [#5890](https://github.com/influxdata/influxdb/issues/5890): Return the time with a selector when there is no group by interval.
- [#6496](https://github.com/influxdata/influxdb/issues/6496): Fix parsing escaped series key when loading database index
- [#6495](https://github.com/influxdata/influxdb/issues/6495): Fix aggregate returns when data is missing from some shards.
- [#6439](https://github.com/influxdata/influxdb/issues/6439): Overwriting points returning old values
- [#6261](https://github.com/influxdata/influxdb/issues/6261): High CPU usage and slow query with DISTINCT

## v0.12.2 [2016-04-20]

### Bugfixes

- [#6271](https://github.com/influxdata/influxdb/issues/6271): Fixed deadlock in tsm1 file store.
- [#6413](https://github.com/influxdata/influxdb/pull/6413): Prevent goroutine leak from persistent http connections. Thanks @aaronknister.
- [#6414](https://github.com/influxdata/influxdb/pull/6414): Send "Connection: close" header for queries.
- [#6419](https://github.com/influxdata/influxdb/issues/6419): Fix panic in transform iterator on division. @thbourlove
- [#6379](https://github.com/influxdata/influxdb/issues/6379): Validate the first argument to percentile() is a variable.
- [#6383](https://github.com/influxdata/influxdb/pull/6383): Recover from a panic during query execution.

## v0.12.1 [2016-04-08]

### Bugfixes

- [#6225](https://github.com/influxdata/influxdb/pull/6225): Refresh admin assets.
- [#6206](https://github.com/influxdata/influxdb/issues/6206): Handle nil values from the tsm1 cursor correctly.
- [#6190](https://github.com/influxdata/influxdb/pull/6190): Fix race on measurementFields.
- [#6248](https://github.com/influxdata/influxdb/issues/6248): Panic using incorrectly quoted "queries" field key.
- [#6257](https://github.com/influxdata/influxdb/issues/6257): CreateShardGroup was incrementing meta data index even when it was idempotent.
- [#6223](https://github.com/influxdata/influxdb/issues/6223): Failure to start/run on Windows. Thanks @mvadu
- [#6229](https://github.com/influxdata/influxdb/issues/6229): Fixed aggregate queries with no GROUP BY to include the end time.


## v0.12.0 [2016-04-05]
### Release Notes
Upgrading to this release requires a little more than just installing the new binary and starting it up. The upgrade process is very quick and should only require a minute of downtime or less. Details on [upgrading to 0.12 are here](https://docs.influxdata.com/influxdb/v0.12/administration/upgrading/).

This release removes all of the old clustering code. It operates as a standalone server. For a free open source HA setup see the [InfluxDB Relay](https://github.com/influxdata/influxdb-relay).

### Features

- [#6012](https://github.com/influxdata/influxdb/pull/6012): Add DROP SHARD support.
- [#6025](https://github.com/influxdata/influxdb/pull/6025): Remove deprecated JSON write path.
- [#5744](https://github.com/influxdata/influxdb/issues/5744): Add integer literal support to the query language.
- [#5939](https://github.com/influxdata/influxdb/issues/5939): Support viewing and killing running queries.
- [#6073](https://github.com/influxdata/influxdb/pull/6073): Iterator stats
- [#6079](https://github.com/influxdata/influxdb/issues/6079): Limit the maximum number of concurrent queries.
- [#6075](https://github.com/influxdata/influxdb/issues/6075): Limit the maximum running time of a query.
- [#6102](https://github.com/influxdata/influxdb/issues/6102): Limit series count in selection
- [#6077](https://github.com/influxdata/influxdb/issues/6077): Limit point count in selection.
- [#6078](https://github.com/influxdata/influxdb/issues/6078): Limit bucket count in selection.
- [#6060](https://github.com/influxdata/influxdb/pull/6060): Add configurable shard duration to retention policies
- [#6116](https://github.com/influxdata/influxdb/pull/6116): Allow `httpd` service to be extensible for routes
- [#6111](https://github.com/influxdata/influxdb/pull/6111): Add ability to build static assest. Improved handling of TAR and ZIP package outputs.
- [#1825](https://github.com/influxdata/influxdb/issues/1825): Implement difference function.
- [#6112](https://github.com/influxdata/influxdb/issues/6112): Implement simple moving average function.
- [#6149](https://github.com/influxdata/influxdb/pull/6149): Kill running queries when server is shutdown.
- [#5372](https://github.com/influxdata/influxdb/pull/5372): Faster shard loading
- [#6148](https://github.com/influxdata/influxdb/pull/6148): Build script is now compatible with Python 3. Added ability to create detached signatures for packages. Build script now uses Python logging facility for messages.
- [#6115](https://github.com/influxdata/influxdb/issues/6115): Support chunking query results mid-series. Limit non-chunked output.
- [#6166](https://github.com/influxdata/influxdb/pull/6166): Teach influxdb client how to use chunked queries and use in the CLI.
- [#6158](https://github.com/influxdata/influxdb/pull/6158): Update influxd to detect an upgrade from `0.11` to `0.12`.  Minor restore bug fixes.
- [#6193](https://github.com/influxdata/influxdb/pull/6193): Fix TypeError when processing empty results in admin UI. Thanks @jonseymour!

### Bugfixes

- [#5152](https://github.com/influxdata/influxdb/issues/5152): Fix where filters when a tag and a filter are combined with OR.
- [#5728](https://github.com/influxdata/influxdb/issues/5728): Properly handle semi-colons as part of the main query loop.
- [#6065](https://github.com/influxdata/influxdb/pull/6065):  Wait for a process termination on influxdb restart @simnv
- [#5252](https://github.com/influxdata/influxdb/issues/5252): Release tarballs contain specific attributes on '.'
- [#5554](https://github.com/influxdata/influxdb/issues/5554): Can't run in alpine linux
- [#6094](https://github.com/influxdata/influxdb/issues/6094): Ensure CREATE RETENTION POLICY and CREATE CONTINUOUS QUERY are idempotent in the correct way.
- [#6061](https://github.com/influxdata/influxdb/issues/6061): [0.12 / master] POST to /write does not write points if request has header 'Content-Type: application/x-www-form-urlencoded'
- [#6140](https://github.com/influxdata/influxdb/issues/6140): Ensure Shard engine not accessed when closed.
- [#6110](https://github.com/influxdata/influxdb/issues/6110): Fix for 0.9 upgrade path when using RPM
- [#6131](https://github.com/influxdata/influxdb/issues/6061): Fix write throughput regression with large number of measurments
- [#6152](https://github.com/influxdata/influxdb/issues/6152): Allow SHARD DURATION to be specified in isolation when creating a database
- [#6153](https://github.com/influxdata/influxdb/issues/6153): Check SHARD DURATION when recreating the same database
- [#6178](https://github.com/influxdata/influxdb/issues/6178): Ensure SHARD DURATION is checked when recreating a retention policy

## v0.11.1 [2016-03-31]

### Bugfixes

- [#6092](https://github.com/influxdata/influxdb/issues/6092): Upgrading directly from 0.9.6.1 to 0.11.0 fails
- [#6129](https://github.com/influxdata/influxdb/pull/6129): Fix default continuous query lease host
- [#6121](https://github.com/influxdata/influxdb/issues/6121): Fix panic: slice index out of bounds in TSM index
- [#6168](https://github.com/influxdata/influxdb/pull/6168): Remove per measurement statsitics
- [#3932](https://github.com/influxdata/influxdb/issues/3932): Invalid timestamp format should throw an error.

## v0.11.0 [2016-03-22]

### Release Notes

There were some important breaking changes in this release. Here's a list of the important things to know before upgrading:

* [SHOW SERIES output has changed](https://github.com/influxdata/influxdb/pull/5937). See [new output in this test diff](https://github.com/influxdata/influxdb/pull/5937/files#diff-0cb24c2b7420b4db507ee3496c371845L263).
* [SHOW TAG VALUES output has changed](https://github.com/influxdata/influxdb/pull/5853)
* JSON write endpoint is disabled by default and will be removed in the next release. You can [turn it back on](https://github.com/influxdata/influxdb/pull/5512) in this release.
* b1/bz1 shards are no longer supported. You must migrate all old shards to TSM using [the migration tool](https://github.com/influxdata/influxdb/blob/master/cmd/influx_tsm/README.md).
* On queries to create databases, retention policies, and users, the default behavior has changed to create `IF NOT EXISTS`. If they already exist, no error will be returned.
* On queries with a selector like `min`, `max`, `first`, and `last` the time returned will be the time for the bucket of the group by window. [Selectors for the time for the specific point](https://github.com/influxdata/influxdb/issues/5926) will be added later.

### Features

- [#5596](https://github.com/influxdata/influxdb/pull/5596): Build improvements for ARM architectures. Also removed `--goarm` and `--pkgarch` build flags.
- [#5541](https://github.com/influxdata/influxdb/pull/5541): Client: Support for adding custom TLS Config for HTTP client.
- [#4299](https://github.com/influxdata/influxdb/pull/4299): Client: Reject uint64 Client.Point.Field values. Thanks @arussellsaw
- [#5550](https://github.com/influxdata/influxdb/pull/5550): Enabled golint for tsdb/engine/wal. @gabelev
- [#5419](https://github.com/influxdata/influxdb/pull/5419): Graphite: Support matching tags multiple times Thanks @m4ce
- [#5598](https://github.com/influxdata/influxdb/pull/5598): Client: Add Ping to v2 client @PSUdaemon
- [#4125](https://github.com/influxdata/influxdb/pull/4125): Admin UI: Fetch and display server version on connect. Thanks @alexiri!
- [#5681](https://github.com/influxdata/influxdb/pull/5681): Stats: Add durations, number currently active to httpd and query executor
- [#5602](https://github.com/influxdata/influxdb/pull/5602): Simplify cluster startup for scripting and deployment
- [#5562](https://github.com/influxdata/influxdb/pull/5562): Graphite: Support matching fields multiple times (@chrusty)
- [#5666](https://github.com/influxdata/influxdb/pull/5666): Manage dependencies with gdm
- [#5512](https://github.com/influxdata/influxdb/pull/5512): HTTP: Add config option to enable HTTP JSON write path which is now disabled by default.
- [#5336](https://github.com/influxdata/influxdb/pull/5366): Enabled golint for influxql. @gabelev
- [#5706](https://github.com/influxdata/influxdb/pull/5706): Cluster setup cleanup
- [#5691](https://github.com/influxdata/influxdb/pull/5691): Remove associated shard data when retention policies are dropped.
- [#5758](https://github.com/influxdata/influxdb/pull/5758): TSM engine stats for cache, WAL, and filestore. Thanks @jonseymour
- [#5844](https://github.com/influxdata/influxdb/pull/5844): Tag TSM engine stats with database and retention policy
- [#5593](https://github.com/influxdata/influxdb/issues/5593): Modify `SHOW TAG VALUES` output for the new query engine to normalize the output.
- [#5862](https://github.com/influxdata/influxdb/pull/5862): Make Admin UI dynamically fetch both client and server versions
- [#2715](https://github.com/influxdata/influxdb/issues/2715): Support using field regex comparisons in the WHERE clause
- [#5994](https://github.com/influxdata/influxdb/issues/5994): Single server
- [#5737](https://github.com/influxdata/influxdb/pull/5737): Admin UI: Display results of multiple queries, not just the first query. Thanks @Vidhuran!
- [#5720](https://github.com/influxdata/influxdb/pull/5720): Admin UI: New button to generate permalink to queries

### Bugfixes

- [#5182](https://github.com/influxdata/influxdb/pull/5182): Graphite: Fix an issue where the default template would be used instead of a more specific one. Thanks @flisky
- [#5489](https://github.com/influxdata/influxdb/pull/5489): Fixes multiple issues causing tests to fail on windows. Thanks @runner-mei
- [#5594](https://github.com/influxdata/influxdb/pull/5594): Fix missing url params on lease redirect - @oldmantaiter
- [#5376](https://github.com/influxdata/influxdb/pull/5376): Fix golint issues in models package. @nuss-justin
- [#5535](https://github.com/influxdata/influxdb/pull/5535): Update README for referring to Collectd
- [#5590](https://github.com/influxdata/influxdb/pull/5590): Fix panic when dropping subscription for unknown retention policy.
- [#5375](https://github.com/influxdata/influxdb/pull/5375): Lint tsdb and tsdb/engine package @nuss-justin
- [#5624](https://github.com/influxdata/influxdb/pull/5624): Fix golint issues in client v2 package @PSUDaemon
- [#5510](https://github.com/influxdata/influxdb/pull/5510): Optimize ReducePercentile @bsideup
- [#5557](https://github.com/influxdata/influxdb/issues/5630): Fixes panic when surrounding the select statement arguments in brackets
- [#5628](https://github.com/influxdata/influxdb/issues/5628): Crashed the server with a bad derivative query
- [#5532](https://github.com/influxdata/influxdb/issues/5532): user passwords not changeable in cluster
- [#5695](https://github.com/influxdata/influxdb/pull/5695): Remove meta servers from node.json
- [#5606](https://github.com/influxdata/influxdb/issues/5606): TSM conversion reproducibly drops data silently
- [#5656](https://github.com/influxdata/influxdb/issues/5656): influx\_tsm: panic during conversion
- [#5696](https://github.com/influxdata/influxdb/issues/5696): Do not drop the database when creating with a retention policy
- [#5724](https://github.com/influxdata/influxdb/issues/5724): influx\_tsm doesn't close file handles properly
- [#5664](https://github.com/influxdata/influxdb/issues/5664): panic in model.Points.scanTo #5664
- [#5716](https://github.com/influxdata/influxdb/pull/5716): models: improve handling of points with empty field names or with no fields.
- [#5719](https://github.com/influxdata/influxdb/issues/5719): Fix cache not deduplicating points
- [#5754](https://github.com/influxdata/influxdb/issues/5754): Adding a node as meta only results in a data node also being registered
- [#5787](https://github.com/influxdata/influxdb/pull/5787): HTTP: Add QueryAuthorizer instance to httpd service’s handler. @chris-ramon
- [#5753](https://github.com/influxdata/influxdb/pull/5753): Ensures that drop-type commands work correctly in a cluster
- [#5814](https://github.com/influxdata/influxdb/issues/5814): Run CQs with the same name from different databases
- [#5699](https://github.com/influxdata/influxdb/issues/5699): Fix potential thread safety issue in cache @jonseymour
- [#5832](https://github.com/influxdata/influxdb/issues/5832): tsm: cache: need to check that snapshot has been sorted @jonseymour
- [#5841](https://github.com/influxdata/influxdb/pull/5841): Reduce tsm allocations by converting time.Time to int64
- [#5842](https://github.com/influxdata/influxdb/issues/5842): Add SeriesList binary marshaling
- [#5854](https://github.com/influxdata/influxdb/issues/5854): failures of tests in tsdb/engine/tsm1 when compiled with go master
- [#5610](https://github.com/influxdata/influxdb/issues/5610): Write into fully-replicated cluster is not replicated across all shards
- [#5880](https://github.com/influxdata/influxdb/issues/5880): TCP connection closed after write (regression/change from 0.9.6)
- [#5865](https://github.com/influxdata/influxdb/issues/5865): Conversion to tsm fails with exceeds max index value
- [#5924](https://github.com/influxdata/influxdb/issues/5924): Missing data after using influx\_tsm
- [#5937](https://github.com/influxdata/influxdb/pull/5937): Rewrite SHOW SERIES to use query engine
- [#5949](https://github.com/influxdata/influxdb/issues/5949): Return error message when improper types are used in SELECT
- [#5963](https://github.com/influxdata/influxdb/pull/5963): Fix possible deadlock
- [#4688](https://github.com/influxdata/influxdb/issues/4688): admin UI doesn't display results for some SHOW queries
- [#6006](https://github.com/influxdata/influxdb/pull/6006): Fix deadlock while running backups
- [#5965](https://github.com/influxdata/influxdb/issues/5965): InfluxDB panic crashes while parsing "-" as Float
- [#5835](https://github.com/influxdata/influxdb/issues/5835): Make CREATE USER default to IF NOT EXISTS
- [#6042](https://github.com/influxdata/influxdb/issues/6042): CreateDatabase failure on Windows, regression from v0.11.0 RC @mvadu
- [#5889](https://github.com/influxdata/influxdb/issues/5889): Fix writing partial TSM index when flush file fails

## v0.10.3 [2016-03-09]

### Bugfixes

- [#5924](https://github.com/influxdata/influxdb/issues/5924): Missing data after using influx\_tsm
- [#5594](https://github.com/influxdata/influxdb/pull/5594): Fix missing url params on lease redirect - @oldmantaiter
- [#5716](https://github.com/influxdata/influxdb/pull/5716): models: improve handling of points with empty field names or with no fields.

## v0.10.2 [2016-03-03]

### Bugfixes

- [#5719](https://github.com/influxdata/influxdb/issues/5719): Fix cache not deduplicating points
- [#5699](https://github.com/influxdata/influxdb/issues/5699): Fix potential thread safety issue in cache @jonseymour
- [#5832](https://github.com/influxdata/influxdb/issues/5832): tsm: cache: need to check that snapshot has been sorted @jonseymour
- [#5857](https://github.com/influxdata/influxdb/issues/5857): panic in tsm1.Values.Deduplicate
- [#5861](https://github.com/influxdata/influxdb/pull/5861): Fix panic when dropping subscription for unknown retention policy.
- [#5880](https://github.com/influxdata/influxdb/issues/5880): TCP connection closed after write (regression/change from 0.9.6)
- [#5865](https://github.com/influxdata/influxdb/issues/5865): Conversion to tsm fails with exceeds max index value

## v0.10.1 [2016-02-18]

### Bugfixes
- [#5696](https://github.com/influxdata/influxdb/issues/5696): Do not drop the database when creating with a retention policy
- [#5724](https://github.com/influxdata/influxdb/issues/5724): influx\_tsm doesn't close file handles properly
- [#5606](https://github.com/influxdata/influxdb/issues/5606): TSM conversion reproducibly drops data silently
- [#5656](https://github.com/influxdata/influxdb/issues/5656): influx\_tsm: panic during conversion
- [#5303](https://github.com/influxdata/influxdb/issues/5303): Protect against stateful mappers returning nothing in the raw executor

## v0.10.0 [2016-02-04]

### Release Notes

This release now uses the TSM storage engine. Old bz1 and b1 shards can still be read, but in a future release you will be required to migrate old shards to TSM. For new shards getting created, or new installations, the TSM storage engine will be used.

This release also changes how clusters are setup. The config file has changed so have a look at the new example. Also, upgrading a single node works, but for upgrading clusters, you'll need help from us. Sent us a note at contact@influxdb.com if you need assistance upgrading a cluster.

### Features
- [#5183](https://github.com/influxdata/influxdb/pull/5183): CLI confirms database exists when USE executed. Thanks @pires
- [#5201](https://github.com/influxdata/influxdb/pull/5201): Allow max UDP buffer size to be configurable. Thanks @sebito91
- [#5194](https://github.com/influxdata/influxdb/pull/5194): Custom continuous query options per query rather than per node.
- [#5224](https://github.com/influxdata/influxdb/pull/5224): Online backup/incremental backup. Restore (for TSM).
- [#5226](https://github.com/influxdata/influxdb/pull/5226): b\*1 to tsm1 shard conversion tool.
- [#5459](https://github.com/influxdata/influxdb/pull/5459): Create `/status` endpoint for health checks.
- [#5460](https://github.com/influxdata/influxdb/pull/5460): Prevent exponential growth in CLI history. Thanks @sczk!
- [#5522](https://github.com/influxdata/influxdb/pull/5522): Optimize tsm1 cache to reduce memory consumption and GC scan time.
- [#5565](https://github.com/influxdata/influxdb/pull/5565): Add configuration for time precision with UDP services. - @tpitale
- [#5226](https://github.com/influxdata/influxdb/pull/5226): b*1 to tsm1 shard conversion tool.

### Bugfixes
- [#5129](https://github.com/influxdata/influxdb/pull/5129): Ensure precision flag is respected by CLI. Thanks @e-dard
- [#5042](https://github.com/influxdata/influxdb/issues/5042): Count with fill(none) will drop 0 valued intervals.
- [#4735](https://github.com/influxdata/influxdb/issues/4735): Fix panic when merging empty results.
- [#5016](https://github.com/influxdata/influxdb/pull/5016): Don't panic if Meta data directory not writable. Thanks @oiooj
- [#5059](https://github.com/influxdata/influxdb/pull/5059): Fix unmarshal of database error by client code. Thanks @farshidtz
- [#4940](https://github.com/influxdata/influxdb/pull/4940): Fix distributed aggregate query query error. Thanks @li-ang
- [#4622](https://github.com/influxdata/influxdb/issues/4622): Fix panic when passing too large of timestamps to OpenTSDB input.
- [#5064](https://github.com/influxdata/influxdb/pull/5064): Full support for parenthesis in SELECT clause, fixes [#5054](https://github.com/influxdata/influxdb/issues/5054). Thanks @mengjinglei
- [#5079](https://github.com/influxdata/influxdb/pull/5079): Ensure tsm WAL encoding buffer can handle large batches.
- [#4303](https://github.com/influxdata/influxdb/issues/4303): Don't drop measurements or series from multiple databases.
- [#5078](https://github.com/influxdata/influxdb/issues/5078): influx non-interactive mode - INSERT must be handled. Thanks @grange74
- [#5178](https://github.com/influxdata/influxdb/pull/5178): SHOW FIELD shouldn't consider VALUES to be valid. Thanks @pires
- [#5158](https://github.com/influxdata/influxdb/pull/5158): Fix panic when writing invalid input to the line protocol.
- [#5264](https://github.com/influxdata/influxdb/pull/5264): Fix panic: runtime error: slice bounds out of range
- [#5186](https://github.com/influxdata/influxdb/pull/5186): Fix database creation with retention statement parsing. Fixes [#5077](https://github.com/influxdata/influxdb/issues/5077). Thanks @pires
- [#5193](https://github.com/influxdata/influxdb/issues/5193): Missing data a minute before current time. Comes back later.
- [#5350](https://github.com/influxdata/influxdb/issues/5350): 'influxd backup' should create backup directory
- [#5262](https://github.com/influxdata/influxdb/issues/5262): Fix a panic when a tag value was empty.
- [#5382](https://github.com/influxdata/influxdb/pull/5382): Fixes some escaping bugs with tag keys and values.
- [#5349](https://github.com/influxdata/influxdb/issues/5349): Validate metadata blob for 'influxd backup'
- [#5469](https://github.com/influxdata/influxdb/issues/5469): Conversion from bz1 to tsm doesn't work as described
- [#5449](https://github.com/influxdata/influxdb/issues/5449): panic when dropping collectd points
- [#5455](https://github.com/influxdata/influxdb/issues/5455): panic: runtime error: slice bounds out of range when loading corrupted wal segment
- [#5478](https://github.com/influxdata/influxdb/issues/5478): panic: interface conversion: interface is float64, not int64
- [#5475](https://github.com/influxdata/influxdb/issues/5475): Ensure appropriate exit code returned for non-interactive use of CLI.
- [#5479](https://github.com/influxdata/influxdb/issues/5479): Bringing up a node as a meta only node causes panic
- [#5504](https://github.com/influxdata/influxdb/issues/5504): create retention policy on unexistant DB crash InfluxDB
- [#5505](https://github.com/influxdata/influxdb/issues/5505): Clear authCache in meta.Client when password changes.
- [#5244](https://github.com/influxdata/influxdb/issues/5244): panic: ensure it's safe to close engine multiple times.

## v0.9.6 [2015-12-09]

### Release Notes
This release has an updated design and implementation of the TSM storage engine. If you had been using tsm1 as your storage engine prior to this release (either 0.9.5.x or 0.9.6 nightly builds) you will have to start with a fresh database.

If you had TSM configuration options set, those have been updated. See the the updated sample configuration for more details: https://github.com/influxdata/influxdb/blob/master/etc/config.sample.toml#L98-L125

### Features
- [#4790](https://github.com/influxdata/influxdb/pull/4790): Allow openTSDB point-level error logging to be disabled
- [#4728](https://github.com/influxdata/influxdb/pull/4728): SHOW SHARD GROUPS. By @mateuszdyminski
- [#4841](https://github.com/influxdata/influxdb/pull/4841): Improve point parsing speed. Lint models pacakge. Thanks @e-dard!
- [#4889](https://github.com/influxdata/influxdb/pull/4889): Implement close notifier and timeout on executors
- [#2676](https://github.com/influxdata/influxdb/issues/2676), [#4866](https://github.com/influxdata/influxdb/pull/4866): Add support for specifying default retention policy in database create. Thanks @pires!
- [#4848](https://github.com/influxdata/influxdb/pull/4848): Added framework for cluster integration testing.
- [#4872](https://github.com/influxdata/influxdb/pull/4872): Add option to disable logging for meta service.
- [#4787](https://github.com/influxdata/influxdb/issues/4787): Now builds on Solaris

### Bugfixes
- [#4849](https://github.com/influxdata/influxdb/issues/4849): Derivative works with count, mean, median, sum, first, last, max, min, and percentile.
- [#4984](https://github.com/influxdata/influxdb/pull/4984): Allow math on fields, fixes regression. Thanks @mengjinglei
- [#4666](https://github.com/influxdata/influxdb/issues/4666): Fix panic in derivative with invalid values.
- [#4404](https://github.com/influxdata/influxdb/issues/4404): Return better error for currently unsupported DELETE queries.
- [#4858](https://github.com/influxdata/influxdb/pull/4858): Validate nested aggregations in queries. Thanks @viru
- [#4921](https://github.com/influxdata/influxdb/pull/4921): Error responses should be JSON-formatted. Thanks @pires
- [#4974](https://github.com/influxdata/influxdb/issues/4974) Fix Data Race in TSDB when setting measurement field name
- [#4876](https://github.com/influxdata/influxdb/pull/4876): Complete lint for monitor and services packages. Thanks @e-dard!
- [#4833](https://github.com/influxdata/influxdb/pull/4833), [#4927](https://github.com/influxdata/influxdb/pull/4927): Fix SHOW MEASURMENTS for clusters. Thanks @li-ang!
- [#4918](https://github.com/influxdata/influxdb/pull/4918): Restore can hang,  Fix [issue #4806](https://github.com/influxdata/influxdb/issues/4806). Thanks @oiooj
- [#4855](https://github.com/influxdata/influxdb/pull/4855): Fix race in TCP proxy shutdown. Thanks @runner-mei!
- [#4411](https://github.com/influxdata/influxdb/pull/4411): Add Access-Control-Expose-Headers to HTTP responses
- [#4768](https://github.com/influxdata/influxdb/pull/4768): CLI history skips blank lines. Thanks @pires
- [#4766](https://github.com/influxdata/influxdb/pull/4766): Update CLI usage output. Thanks @aneshas
- [#4804](https://github.com/influxdata/influxdb/pull/4804): Complete lint for services/admin. Thanks @nii236
- [#4796](https://github.com/influxdata/influxdb/pull/4796): Check point without fields. Thanks @CrazyJvm
- [#4815](https://github.com/influxdata/influxdb/pull/4815): Added `Time` field into aggregate output across the cluster. Thanks @li-ang
- [#4817](https://github.com/influxdata/influxdb/pull/4817): Fix Min,Max,Top,Bottom function when query distributed node. Thanks @mengjinglei
- [#4878](https://github.com/influxdata/influxdb/pull/4878): Fix String() function for several InfluxQL statement types
- [#4913](https://github.com/influxdata/influxdb/pull/4913): Fix b1 flush deadlock
- [#3170](https://github.com/influxdata/influxdb/issues/3170), [#4921](https://github.com/influxdata/influxdb/pull/4921): Database does not exist error is now JSON. Thanks @pires!
- [#5029](https://github.com/influxdata/influxdb/pull/5029): Drop UDP point on bad parse.

## v0.9.5 [2015-11-20]

### Release Notes

- Field names for the internal stats have been changed to be more inline with Go style.
- 0.9.5 is reverting to Go 1.4.2 due to unresolved issues with Go 1.5.1.

There are breaking changes in this release:
- The filesystem hierarchy for packages has been changed, namely:
  - Binaries are now located in `/usr/bin` (previously `/opt/influxdb`)
  - Configuration files are now located in `/etc/influxdb` (previously `/etc/opt/influxdb`)
  - Data directories are now located in `/var/lib/influxdb` (previously `/var/opt/influxdb`)
  - Scripts are now located in `/usr/lib/influxdb/scripts` (previously `/opt/influxdb`)

### Features
- [#4702](https://github.com/influxdata/influxdb/pull/4702): Support 'history' command at CLI
- [#4098](https://github.com/influxdata/influxdb/issues/4098): Enable `golint` on the code base - uuid subpackage
- [#4141](https://github.com/influxdata/influxdb/pull/4141): Control whether each query should be logged
- [#4065](https://github.com/influxdata/influxdb/pull/4065): Added precision support in cmd client. Thanks @sbouchex
- [#4140](https://github.com/influxdata/influxdb/pull/4140): Make storage engine configurable
- [#4161](https://github.com/influxdata/influxdb/pull/4161): Implement bottom selector function
- [#4204](https://github.com/influxdata/influxdb/pull/4204): Allow module-level selection for SHOW STATS
- [#4208](https://github.com/influxdata/influxdb/pull/4208): Allow module-level selection for SHOW DIAGNOSTICS
- [#4196](https://github.com/influxdata/influxdb/pull/4196): Export tsdb.Iterator
- [#4198](https://github.com/influxdata/influxdb/pull/4198): Add basic cluster-service stats
- [#4262](https://github.com/influxdata/influxdb/pull/4262): Allow configuration of UDP retention policy
- [#4265](https://github.com/influxdata/influxdb/pull/4265): Add statistics for Hinted-Handoff
- [#4284](https://github.com/influxdata/influxdb/pull/4284): Add exponential backoff for hinted-handoff failures
- [#4310](https://github.com/influxdata/influxdb/pull/4310): Support dropping non-Raft nodes. Work mostly by @corylanou
- [#4348](https://github.com/influxdata/influxdb/pull/4348): Public ApplyTemplate function for graphite parser.
- [#4178](https://github.com/influxdata/influxdb/pull/4178): Support fields in graphite parser. Thanks @roobert!
- [#4409](https://github.com/influxdata/influxdb/pull/4409): wire up INTO queries.
- [#4379](https://github.com/influxdata/influxdb/pull/4379): Auto-create database for UDP input.
- [#4375](https://github.com/influxdata/influxdb/pull/4375): Add Subscriptions so data can be 'forked' out of InfluxDB to another third party.
- [#4506](https://github.com/influxdata/influxdb/pull/4506): Register with Enterprise service and upload stats, if token is available.
- [#4516](https://github.com/influxdata/influxdb/pull/4516): Hinted-handoff refactor, with new statistics and diagnostics
- [#4501](https://github.com/influxdata/influxdb/pull/4501): Allow filtering SHOW MEASUREMENTS by regex.
- [#4547](https://github.com/influxdata/influxdb/pull/4547): Allow any node to be dropped, even a raft node (even the leader).
- [#4600](https://github.com/influxdata/influxdb/pull/4600): ping endpoint can wait for leader
- [#4648](https://github.com/influxdata/influxdb/pull/4648): UDP Client (v2 client)
- [#4690](https://github.com/influxdata/influxdb/pull/4690): SHOW SHARDS now includes database and policy. Thanks @pires
- [#4676](https://github.com/influxdata/influxdb/pull/4676): UDP service listener performance enhancements
- [#4659](https://github.com/influxdata/influxdb/pull/4659): Support IF EXISTS for DROP DATABASE. Thanks @ch33hau
- [#4721](https://github.com/influxdata/influxdb/pull/4721): Export tsdb.InterfaceValues
- [#4681](https://github.com/influxdata/influxdb/pull/4681): Increase default buffer size for collectd and graphite listeners
- [#4685](https://github.com/influxdata/influxdb/pull/4685): Automatically promote node to raft peer if drop server results in removing a raft peer.
- [#4846](https://github.com/influxdata/influxdb/pull/4846): Allow NaN as a valid value on the graphite service; discard these points silently (graphite compatibility). Thanks @jsternberg!

### Bugfixes
- [#4193](https://github.com/influxdata/influxdb/issues/4193): Less than or equal to inequality is not inclusive for time in where clause
- [#4235](https://github.com/influxdata/influxdb/issues/4235): "ORDER BY DESC" doesn't properly order
- [#4789](https://github.com/influxdata/influxdb/pull/4789): Decode WHERE fields during aggregates. Fix [issue #4701](https://github.com/influxdata/influxdb/issues/4701).
- [#4778](https://github.com/influxdata/influxdb/pull/4778): If there are no points to count, count is 0.
- [#4715](https://github.com/influxdata/influxdb/pull/4715): Fix panic during Raft-close. Fix [issue #4707](https://github.com/influxdata/influxdb/issues/4707). Thanks @oiooj
- [#4643](https://github.com/influxdata/influxdb/pull/4643): Fix panic during backup restoration. Thanks @oiooj
- [#4632](https://github.com/influxdata/influxdb/pull/4632): Fix parsing of IPv6 hosts in client package. Thanks @miguelxpn
- [#4389](https://github.com/influxdata/influxdb/pull/4389): Don't add a new segment file on each hinted-handoff purge cycle.
- [#4166](https://github.com/influxdata/influxdb/pull/4166): Fix parser error on invalid SHOW
- [#3457](https://github.com/influxdata/influxdb/issues/3457): [0.9.3] cannot select field names with prefix + "." that match the measurement name
- [#4704](https://github.com/influxdata/influxdb/pull/4704). Tighten up command parsing within CLI. Thanks @pires
- [#4225](https://github.com/influxdata/influxdb/pull/4225): Always display diags in name-sorted order
- [#4111](https://github.com/influxdata/influxdb/pull/4111): Update pre-commit hook for go vet composites
- [#4136](https://github.com/influxdata/influxdb/pull/4136): Return an error-on-write if target retention policy does not exist. Thanks for the report @ymettier
- [#4228](https://github.com/influxdata/influxdb/pull/4228): Add build timestamp to version information.
- [#4124](https://github.com/influxdata/influxdb/issues/4124): Missing defer/recover/panic idiom in HTTPD service
- [#4238](https://github.com/influxdata/influxdb/pull/4238): Fully disable hinted-handoff service if so requested.
- [#4165](https://github.com/influxdata/influxdb/pull/4165): Tag all Go runtime stats when writing to internal database.
- [#4586](https://github.com/influxdata/influxdb/pull/4586): Exit when invalid engine is selected
- [#4118](https://github.com/influxdata/influxdb/issues/4118): Return consistent, correct result for SHOW MEASUREMENTS with multiple AND conditions
- [#4191](https://github.com/influxdata/influxdb/pull/4191): Correctly marshal remote mapper responses. Fixes [#4170](https://github.com/influxdata/influxdb/issues/4170)
- [#4222](https://github.com/influxdata/influxdb/pull/4222): Graphite TCP connections should not block shutdown
- [#4180](https://github.com/influxdata/influxdb/pull/4180): Cursor & SelectMapper Refactor
- [#1577](https://github.com/influxdata/influxdb/issues/1577): selectors (e.g. min, max, first, last) should have equivalents to return the actual point
- [#4264](https://github.com/influxdata/influxdb/issues/4264): Refactor map functions to use list of values
- [#4278](https://github.com/influxdata/influxdb/pull/4278): Fix error marshalling across the cluster
- [#4149](https://github.com/influxdata/influxdb/pull/4149): Fix derivative unnecessarily requires aggregate function.  Thanks @peekeri!
- [#4674](https://github.com/influxdata/influxdb/pull/4674): Fix panic during restore. Thanks @simcap.
- [#4725](https://github.com/influxdata/influxdb/pull/4725): Don't list deleted shards during SHOW SHARDS.
- [#4237](https://github.com/influxdata/influxdb/issues/4237): DERIVATIVE() edge conditions
- [#4263](https://github.com/influxdata/influxdb/issues/4263): derivative does not work when data is missing
- [#4293](https://github.com/influxdata/influxdb/pull/4293): Ensure shell is invoked when touching PID file. Thanks @christopherjdickson
- [#4296](https://github.com/influxdata/influxdb/pull/4296): Reject line protocol ending with '-'. Fixes [#4272](https://github.com/influxdata/influxdb/issues/4272)
- [#4333](https://github.com/influxdata/influxdb/pull/4333): Retry monitor storage creation and storage only on Leader.
- [#4276](https://github.com/influxdata/influxdb/issues/4276): Walk DropSeriesStatement & check for empty sources
- [#4465](https://github.com/influxdata/influxdb/pull/4465): Actually display a message if the CLI can't connect to the database.
- [#4342](https://github.com/influxdata/influxdb/pull/4342): Fix mixing aggregates and math with non-aggregates. Thanks @kostya-sh.
- [#4349](https://github.com/influxdata/influxdb/issues/4349): If HH can't unmarshal a block, skip that block.
- [#4502](https://github.com/influxdata/influxdb/pull/4502): Don't crash on Graphite close, if Graphite not fully open. Thanks for the report @ranjib
- [#4354](https://github.com/influxdata/influxdb/pull/4353): Fully lock node queues during hinted handoff. Fixes one cause of missing data on clusters.
- [#4357](https://github.com/influxdata/influxdb/issues/4357): Fix similar float values encoding overflow Thanks @dgryski!
- [#4344](https://github.com/influxdata/influxdb/issues/4344): Make client.Write default to client.precision if none is given.
- [#3429](https://github.com/influxdata/influxdb/issues/3429): Incorrect parsing of regex containing '/'
- [#4374](https://github.com/influxdata/influxdb/issues/4374): Add tsm1 quickcheck tests
- [#4644](https://github.com/influxdata/influxdb/pull/4644): Check for response errors during token check, fixes issue [#4641](https://github.com/influxdata/influxdb/issues/4641)
- [#4377](https://github.com/influxdata/influxdb/pull/4377): Hinted handoff should not process dropped nodes
- [#4365](https://github.com/influxdata/influxdb/issues/4365): Prevent panic in DecodeSameTypeBlock
- [#4280](https://github.com/influxdata/influxdb/issues/4280): Only drop points matching WHERE clause
- [#4443](https://github.com/influxdata/influxdb/pull/4443): Fix race condition while listing store's shards. Fixes [#4442](https://github.com/influxdata/influxdb/issues/4442)
- [#4410](https://github.com/influxdata/influxdb/pull/4410): Fix infinite recursion in statement string(). Thanks @kostya-sh
- [#4360](https://github.com/influxdata/influxdb/issues/4360): Aggregate Selectors overwrite values during post-processing
- [#4421](https://github.com/influxdata/influxdb/issues/4421): Fix line protocol accepting tags with no values
- [#4434](https://github.com/influxdata/influxdb/pull/4434): Allow 'E' for scientific values. Fixes [#4433](https://github.com/influxdata/influxdb/issues/4433)
- [#4431](https://github.com/influxdata/influxdb/issues/4431): Add tsm1 WAL QuickCheck
- [#4438](https://github.com/influxdata/influxdb/pull/4438): openTSDB service shutdown fixes
- [#4447](https://github.com/influxdata/influxdb/pull/4447): Fixes to logrotate file. Thanks @linsomniac.
- [#3820](https://github.com/influxdata/influxdb/issues/3820): Fix js error in admin UI.
- [#4460](https://github.com/influxdata/influxdb/issues/4460): tsm1 meta lint
- [#4415](https://github.com/influxdata/influxdb/issues/4415): Selector (like max, min, first, etc) return a string instead of timestamp
- [#4472](https://github.com/influxdata/influxdb/issues/4472): Fix 'too many points in GROUP BY interval' error
- [#4475](https://github.com/influxdata/influxdb/issues/4475): Fix SHOW TAG VALUES error message.
- [#4486](https://github.com/influxdata/influxdb/pull/4486): Fix missing comments for runner package
- [#4497](https://github.com/influxdata/influxdb/pull/4497): Fix sequence in meta proto
- [#3367](https://github.com/influxdata/influxdb/issues/3367): Negative timestamps are parsed correctly by the line protocol.
- [#4563](https://github.com/influxdata/influxdb/pull/4536): Fix broken subscriptions updates.
- [#4538](https://github.com/influxdata/influxdb/issues/4538): Dropping database under a write load causes panics
- [#4582](https://github.com/influxdata/influxdb/pull/4582): Correct logging tags in cluster and TCP package. Thanks @oiooj
- [#4513](https://github.com/influxdata/influxdb/issues/4513): TSM1: panic: runtime error: index out of range
- [#4521](https://github.com/influxdata/influxdb/issues/4521): TSM1: panic: decode of short block: got 1, exp 9
- [#4587](https://github.com/influxdata/influxdb/pull/4587): Prevent NaN float values from being stored
- [#4596](https://github.com/influxdata/influxdb/pull/4596): Skip empty string for start position when parsing line protocol @Thanks @ch33hau
- [#4610](https://github.com/influxdata/influxdb/pull/4610): Make internal stats names consistent with Go style.
- [#4625](https://github.com/influxdata/influxdb/pull/4625): Correctly handle bad write requests. Thanks @oiooj.
- [#4650](https://github.com/influxdata/influxdb/issues/4650): Importer should skip empty lines
- [#4651](https://github.com/influxdata/influxdb/issues/4651): Importer doesn't flush out last batch
- [#4602](https://github.com/influxdata/influxdb/issues/4602): Fixes data race between PointsWriter and Subscriber services.
- [#4691](https://github.com/influxdata/influxdb/issues/4691): Enable toml test `TestConfig_Encode`.
- [#4283](https://github.com/influxdata/influxdb/pull/4283): Disable HintedHandoff if configuration is not set.
- [#4703](https://github.com/influxdata/influxdb/pull/4703): Complete lint for cmd/influx. Thanks @pablolmiranda

## v0.9.4 [2015-09-14]

### Release Notes
With this release InfluxDB is moving to Go 1.5.

### Features
- [#4050](https://github.com/influxdata/influxdb/pull/4050): Add stats to collectd
- [#3771](https://github.com/influxdata/influxdb/pull/3771): Close idle Graphite TCP connections
- [#3755](https://github.com/influxdata/influxdb/issues/3755): Add option to build script. Thanks @fg2it
- [#3863](https://github.com/influxdata/influxdb/pull/3863): Move to Go 1.5
- [#3892](https://github.com/influxdata/influxdb/pull/3892): Support IF NOT EXISTS for CREATE DATABASE
- [#3916](https://github.com/influxdata/influxdb/pull/3916): New statistics and diagnostics support. Graphite first to be instrumented.
- [#3901](https://github.com/influxdata/influxdb/pull/3901): Add consistency level option to influx cli Thanks @takayuki
- [#4048](https://github.com/influxdata/influxdb/pull/4048): Add statistics to Continuous Query service
- [#4049](https://github.com/influxdata/influxdb/pull/4049): Add stats to the UDP input
- [#3876](https://github.com/influxdata/influxdb/pull/3876): Allow the following syntax in CQs: INTO "1hPolicy".:MEASUREMENT
- [#3975](https://github.com/influxdata/influxdb/pull/3975): Add shard copy service
- [#3986](https://github.com/influxdata/influxdb/pull/3986): Support sorting by time desc
- [#3930](https://github.com/influxdata/influxdb/pull/3930): Wire up TOP aggregate function - fixes [#1821](https://github.com/influxdata/influxdb/issues/1821)
- [#4045](https://github.com/influxdata/influxdb/pull/4045): Instrument cluster-level points writer
- [#3996](https://github.com/influxdata/influxdb/pull/3996): Add statistics to httpd package
- [#4003](https://github.com/influxdata/influxdb/pull/4033): Add logrotate configuration.
- [#4043](https://github.com/influxdata/influxdb/pull/4043): Add stats and batching to openTSDB input
- [#4042](https://github.com/influxdata/influxdb/pull/4042): Add pending batches control to batcher
- [#4006](https://github.com/influxdata/influxdb/pull/4006): Add basic statistics for shards
- [#4072](https://github.com/influxdata/influxdb/pull/4072): Add statistics for the WAL.

### Bugfixes
- [#4042](https://github.com/influxdata/influxdb/pull/4042): Set UDP input batching defaults as needed.
- [#3785](https://github.com/influxdata/influxdb/issues/3785): Invalid time stamp in graphite metric causes panic
- [#3804](https://github.com/influxdata/influxdb/pull/3804): init.d script fixes, fixes issue 3803.
- [#3823](https://github.com/influxdata/influxdb/pull/3823): Deterministic ordering for first() and last()
- [#3869](https://github.com/influxdata/influxdb/issues/3869): Seemingly deadlocked when ingesting metrics via graphite plugin
- [#3856](https://github.com/influxdata/influxdb/pull/3856): Minor changes to retention enforcement.
- [#3884](https://github.com/influxdata/influxdb/pull/3884): Fix two panics in WAL that can happen at server startup
- [#3868](https://github.com/influxdata/influxdb/pull/3868): Add shell option to start the daemon on CentOS. Thanks @SwannCroiset.
- [#3886](https://github.com/influxdata/influxdb/pull/3886): Prevent write timeouts due to lock contention in WAL
- [#3574](https://github.com/influxdata/influxdb/issues/3574): Querying data node causes panic
- [#3913](https://github.com/influxdata/influxdb/issues/3913): Convert meta shard owners to objects
- [#4026](https://github.com/influxdata/influxdb/pull/4026): Support multiple Graphite inputs. Fixes issue [#3636](https://github.com/influxdata/influxdb/issues/3636)
- [#3927](https://github.com/influxdata/influxdb/issues/3927): Add WAL lock to prevent timing lock contention
- [#3928](https://github.com/influxdata/influxdb/issues/3928): Write fails for multiple points when tag starts with quote
- [#3901](https://github.com/influxdata/influxdb/pull/3901): Unblock relaxed write consistency level Thanks @takayuki!
- [#3950](https://github.com/influxdata/influxdb/pull/3950): Limit bz1 quickcheck tests to 10 iterations on CI
- [#3977](https://github.com/influxdata/influxdb/pull/3977): Silence wal logging during testing
- [#3931](https://github.com/influxdata/influxdb/pull/3931): Don't precreate shard groups entirely in the past
- [#3960](https://github.com/influxdata/influxdb/issues/3960): possible "catch up" bug with nodes down in a cluster
- [#3980](https://github.com/influxdata/influxdb/pull/3980): 'service stop' waits until service actually stops. Fixes issue #3548.
- [#4016](https://github.com/influxdata/influxdb/pull/4016): Shutdown Graphite UDP on SIGTERM.
- [#4034](https://github.com/influxdata/influxdb/pull/4034): Rollback bolt tx on mapper open error
- [#3848](https://github.com/influxdata/influxdb/issues/3848): restart influxdb causing panic
- [#3881](https://github.com/influxdata/influxdb/issues/3881): panic: runtime error: invalid memory address or nil pointer dereference
- [#3926](https://github.com/influxdata/influxdb/issues/3926): First or last value of `GROUP BY time(x)` is often null. Fixed by [#4038](https://github.com/influxdata/influxdb/pull/4038)
- [#4053](https://github.com/influxdata/influxdb/pull/4053): Prohibit dropping default retention policy.
- [#4060](https://github.com/influxdata/influxdb/pull/4060): Don't log EOF error in openTSDB input.
- [#3978](https://github.com/influxdata/influxdb/issues/3978): [0.9.3] (regression) cannot use GROUP BY * with more than a single field in SELECT clause
- [#4058](https://github.com/influxdata/influxdb/pull/4058): Disable bz1 recompression
- [#3902](https://github.com/influxdata/influxdb/issues/3902): [0.9.3] DB should not crash when using invalid expression "GROUP BY time"
- [#3718](https://github.com/influxdata/influxdb/issues/3718): Derivative query with group by time but no aggregate function should fail parse

## v0.9.3 [2015-08-26]

### Release Notes

There are breaking changes in this release.
 - To store data points as integers you must now append `i` to the number if using the line protocol.
 - If you have a UDP input configured, you should check the UDP section of [the new sample configuration file](https://github.com/influxdata/influxdb/blob/master/etc/config.sample.toml) to learn how to modify existing configuration files, as 0.9.3 now expects multiple UDP inputs.
 - Configuration files must now have an entry for `wal-dir` in the `[data]` section. Check [new sample configuration file](https://github.com/influxdata/influxdb/blob/master/etc/config.sample.toml) for more details.
 - The implicit `GROUP BY *` that was added to every `SELECT *` has been removed. Instead any tags in the data are now part of the columns in the returned query.

Please see the *Features* section below for full details.

### Features
- [#3376](https://github.com/influxdata/influxdb/pull/3376): Support for remote shard query mapping
- [#3372](https://github.com/influxdata/influxdb/pull/3372): Support joining nodes to existing cluster
- [#3426](https://github.com/influxdata/influxdb/pull/3426): Additional logging for continuous queries. Thanks @jhorwit2
- [#3478](https://github.com/influxdata/influxdb/pull/3478): Support incremental cluster joins
- [#3519](https://github.com/influxdata/influxdb/pull/3519): **--BREAKING CHANGE--** Update line protocol to require trailing i for field values that are integers
- [#3529](https://github.com/influxdata/influxdb/pull/3529): Add TLS support for OpenTSDB plugin. Thanks @nathanielc
- [#3421](https://github.com/influxdata/influxdb/issues/3421): Should update metastore and cluster if IP or hostname changes
- [#3502](https://github.com/influxdata/influxdb/pull/3502): Importer for 0.8.9 data via the CLI
- [#3564](https://github.com/influxdata/influxdb/pull/3564): Fix alias, maintain column sort order
- [#3585](https://github.com/influxdata/influxdb/pull/3585): Additional test coverage for non-existent fields
- [#3246](https://github.com/influxdata/influxdb/issues/3246): Allow overriding of configuration parameters using environment variables
- [#3599](https://github.com/influxdata/influxdb/pull/3599): **--BREAKING CHANGE--** Support multiple UDP inputs. Thanks @tpitale
- [#3636](https://github.com/influxdata/influxdb/pull/3639): Cap auto-created retention policy replica count at 3
- [#3641](https://github.com/influxdata/influxdb/pull/3641): Logging enhancements and single-node rename
- [#3635](https://github.com/influxdata/influxdb/pull/3635): Add build branch to version output.
- [#3115](https://github.com/influxdata/influxdb/pull/3115): Various init.d script improvements. Thanks @KoeSystems.
- [#3628](https://github.com/influxdata/influxdb/pull/3628): Wildcard expansion of tags and fields for raw queries
- [#3721](https://github.com/influxdata/influxdb/pull/3721): interpret number literals compared against time as nanoseconds from epoch
- [#3514](https://github.com/influxdata/influxdb/issues/3514): Implement WAL outside BoltDB with compaction
- [#3544](https://github.com/influxdata/influxdb/pull/3544): Implement compression on top of BoltDB
- [#3795](https://github.com/influxdata/influxdb/pull/3795): Throttle import
- [#3584](https://github.com/influxdata/influxdb/pull/3584): Import/export documenation

### Bugfixes
- [#3405](https://github.com/influxdata/influxdb/pull/3405): Prevent database panic when fields are missing. Thanks @jhorwit2
- [#3411](https://github.com/influxdata/influxdb/issues/3411): 500 timeout on write
- [#3420](https://github.com/influxdata/influxdb/pull/3420): Catch opentsdb malformed tags. Thanks @nathanielc.
- [#3404](https://github.com/influxdata/influxdb/pull/3404): Added support for escaped single quotes in query string. Thanks @jhorwit2
- [#3414](https://github.com/influxdata/influxdb/issues/3414): Shard mappers perform query re-writing
- [#3525](https://github.com/influxdata/influxdb/pull/3525): check if fields are valid during parse time.
- [#3511](https://github.com/influxdata/influxdb/issues/3511): Sending a large number of tag causes panic
- [#3288](https://github.com/influxdata/influxdb/issues/3288): Run go fuzz on the line-protocol input
- [#3545](https://github.com/influxdata/influxdb/issues/3545): Fix parsing string fields with newlines
- [#3579](https://github.com/influxdata/influxdb/issues/3579): Revert breaking change to `client.NewClient` function
- [#3580](https://github.com/influxdata/influxdb/issues/3580): Do not allow wildcards with fields in select statements
- [#3530](https://github.com/influxdata/influxdb/pull/3530): Aliasing a column no longer works
- [#3436](https://github.com/influxdata/influxdb/issues/3436): Fix panic in hinted handoff queue processor
- [#3401](https://github.com/influxdata/influxdb/issues/3401): Derivative on non-numeric fields panics db
- [#3583](https://github.com/influxdata/influxdb/issues/3583): Inserting value in scientific notation with a trailing i causes panic
- [#3611](https://github.com/influxdata/influxdb/pull/3611): Fix query arithmetic with integers
- [#3326](https://github.com/influxdata/influxdb/issues/3326): simple regex query fails with cryptic error
- [#3618](https://github.com/influxdata/influxdb/pull/3618): Fix collectd stats panic on i386. Thanks @richterger
- [#3625](https://github.com/influxdata/influxdb/pull/3625): Don't panic when aggregate and raw queries are in a single statement
- [#3629](https://github.com/influxdata/influxdb/pull/3629): Use sensible batching defaults for Graphite.
- [#3638](https://github.com/influxdata/influxdb/pull/3638): Cluster config fixes and removal of meta.peers config field
- [#3640](https://github.com/influxdata/influxdb/pull/3640): Shutdown Graphite service when signal received.
- [#3632](https://github.com/influxdata/influxdb/issues/3632): Make single-node host renames more seamless
- [#3656](https://github.com/influxdata/influxdb/issues/3656): Silence snapshotter logger for testing
- [#3651](https://github.com/influxdata/influxdb/pull/3651): Fully remove series when dropped.
- [#3517](https://github.com/influxdata/influxdb/pull/3517): Batch CQ writes to avoid timeouts. Thanks @dim.
- [#3522](https://github.com/influxdata/influxdb/pull/3522): Consume CQ results on request timeouts. Thanks @dim.
- [#3646](https://github.com/influxdata/influxdb/pull/3646): Fix nil FieldCodec panic.
- [#3672](https://github.com/influxdata/influxdb/pull/3672): Reduce in-memory index by 20%-30%
- [#3673](https://github.com/influxdata/influxdb/pull/3673): Improve query performance by removing unnecessary tagset sorting.
- [#3676](https://github.com/influxdata/influxdb/pull/3676): Improve query performance by memomizing mapper output keys.
- [#3686](https://github.com/influxdata/influxdb/pull/3686): Ensure 'p' parameter is not logged, even on OPTIONS requests.
- [#3687](https://github.com/influxdata/influxdb/issues/3687): Fix panic: runtime error: makeslice: len out of range in hinted handoff
- [#3697](https://github.com/influxdata/influxdb/issues/3697):  Correctly merge non-chunked results for same series. Fix issue #3242.
- [#3708](https://github.com/influxdata/influxdb/issues/3708): Fix double escaping measurement name during cluster replication
- [#3704](https://github.com/influxdata/influxdb/issues/3704): cluster replication issue for measurement name containing backslash
- [#3681](https://github.com/influxdata/influxdb/issues/3681): Quoted measurement names fail
- [#3681](https://github.com/influxdata/influxdb/issues/3682): Fix inserting string value with backslashes
- [#3735](https://github.com/influxdata/influxdb/issues/3735): Append to small bz1 blocks
- [#3736](https://github.com/influxdata/influxdb/pull/3736): Update shard group duration with retention policy changes. Thanks for the report @papylhomme
- [#3539](https://github.com/influxdata/influxdb/issues/3539): parser incorrectly accepts NaN as numerical value, but not always
- [#3790](https://github.com/influxdata/influxdb/pull/3790): Fix line protocol parsing equals in measurements and NaN values
- [#3778](https://github.com/influxdata/influxdb/pull/3778): Don't panic if SELECT on time.
- [#3824](https://github.com/influxdata/influxdb/issues/3824): tsdb.Point.MarshalBinary needs to support all number types
- [#3828](https://github.com/influxdata/influxdb/pull/3828): Support all number types when decoding a point
- [#3853](https://github.com/influxdata/influxdb/pull/3853): Use 4KB default block size for bz1
- [#3607](https://github.com/influxdata/influxdb/issues/3607): Fix unable to query influxdb due to deadlock in metastore.  Thanks @ccutrer!

## v0.9.2 [2015-07-24]

### Features
- [#3177](https://github.com/influxdata/influxdb/pull/3177): Client supports making HTTPS requests. Thanks @jipperinbham
- [#3299](https://github.com/influxdata/influxdb/pull/3299): Refactor query engine for distributed query support.
- [#3334](https://github.com/influxdata/influxdb/pull/3334): Clean shutdown of influxd. Thanks @mcastilho

### Bugfixes

- [#3180](https://github.com/influxdata/influxdb/pull/3180): Log GOMAXPROCS, version, and commit on startup.
- [#3218](https://github.com/influxdata/influxdb/pull/3218): Allow write timeouts to be configurable.
- [#3184](https://github.com/influxdata/influxdb/pull/3184): Support basic auth in admin interface. Thanks @jipperinbham!
- [#3236](https://github.com/influxdata/influxdb/pull/3236): Fix display issues in admin interface.
- [#3232](https://github.com/influxdata/influxdb/pull/3232): Set logging prefix for metastore.
- [#3230](https://github.com/influxdata/influxdb/issues/3230): panic: unable to parse bool value
- [#3245](https://github.com/influxdata/influxdb/issues/3245): Error using graphite plugin with multiple filters
- [#3223](https://github.com/influxdata/influxdb/issues/323): default graphite template cannot have extra tags
- [#3255](https://github.com/influxdata/influxdb/pull/3255): Flush WAL on start-up as soon as possible.
- [#3289](https://github.com/influxdata/influxdb/issues/3289): InfluxDB crashes on floats without decimal
- [#3298](https://github.com/influxdata/influxdb/pull/3298): Corrected WAL & flush parameters in default config. Thanks @jhorwit2
- [#3152](https://github.com/influxdata/influxdb/issues/3159): High CPU Usage with unsorted writes
- [#3307](https://github.com/influxdata/influxdb/pull/3307): Fix regression parsing boolean values True/False
- [#3304](https://github.com/influxdata/influxdb/pull/3304): Fixed httpd logger to log user from query params. Thanks @jhorwit2
- [#3332](https://github.com/influxdata/influxdb/pull/3332): Add SLIMIT and SOFFSET to string version of AST.
- [#3335](https://github.com/influxdata/influxdb/pull/3335): Don't drop all data on DROP DATABASE. Thanks to @PierreF for the report
- [#2761](https://github.com/influxdata/influxdb/issues/2761): Make SHOW RETENTION POLICIES consistent with other queries.
- [#3356](https://github.com/influxdata/influxdb/pull/3356): Disregard semicolons after database name in use command. Thanks @timraymond.
- [#3351](https://github.com/influxdata/influxdb/pull/3351): Handle malformed regex comparisons during parsing. Thanks @rnubel
- [#3244](https://github.com/influxdata/influxdb/pull/3244): Wire up admin privilege grant and revoke.
- [#3259](https://github.com/influxdata/influxdb/issues/3259): Respect privileges for queries.
- [#3256](https://github.com/influxdata/influxdb/pull/3256): Remove unnecessary timeout in WaitForLeader(). Thanks @cannium.
- [#3380](https://github.com/influxdata/influxdb/issues/3380): Parser fix, only allow ORDER BY ASC and ORDER BY time ASC.
- [#3319](https://github.com/influxdata/influxdb/issues/3319): restarting process irrevocably BREAKS measurements with spaces
- [#3453](https://github.com/influxdata/influxdb/issues/3453): Remove outdated `dump` command from CLI.
- [#3463](https://github.com/influxdata/influxdb/issues/3463): Fix aggregate queries and time precision on where clauses.

## v0.9.1 [2015-07-02]

### Features

- [2650](https://github.com/influxdata/influxdb/pull/2650): Add SHOW GRANTS FOR USER statement. Thanks @n1tr0g.
- [3125](https://github.com/influxdata/influxdb/pull/3125): Graphite Input Protocol Parsing
- [2746](https://github.com/influxdata/influxdb/pull/2746): New Admin UI/interface
- [3036](https://github.com/influxdata/influxdb/pull/3036): Write Ahead Log (WAL)
- [3014](https://github.com/influxdata/influxdb/issues/3014): Implement Raft snapshots

### Bugfixes

- [3013](https://github.com/influxdata/influxdb/issues/3013): Panic error with inserting values with commas
- [#2956](https://github.com/influxdata/influxdb/issues/2956): Type mismatch in derivative
- [#2908](https://github.com/influxdata/influxdb/issues/2908): Field mismatch error messages need to be updated
- [#2931](https://github.com/influxdata/influxdb/pull/2931): Services and reporting should wait until cluster has leader.
- [#2943](https://github.com/influxdata/influxdb/issues/2943): Ensure default retention policies are fully replicated
- [#2948](https://github.com/influxdata/influxdb/issues/2948): Field mismatch error message to include measurement name
- [#2919](https://github.com/influxdata/influxdb/issues/2919): Unable to insert negative floats
- [#2935](https://github.com/influxdata/influxdb/issues/2935): Hook CPU and memory profiling back up.
- [#2960](https://github.com/influxdata/influxdb/issues/2960): Cluster Write Errors.
- [#2928](https://github.com/influxdata/influxdb/pull/2928): Start work to set InfluxDB version in HTTP response headers. Thanks @neonstalwart.
- [#2969](https://github.com/influxdata/influxdb/pull/2969): Actually set HTTP version in responses.
- [#2993](https://github.com/influxdata/influxdb/pull/2993): Don't log each UDP batch.
- [#2994](https://github.com/influxdata/influxdb/pull/2994): Don't panic during wilcard expansion if no default database specified.
- [#3002](https://github.com/influxdata/influxdb/pull/3002): Remove measurement from shard's index on DROP MEASUREMENT.
- [#3021](https://github.com/influxdata/influxdb/pull/3021): Correct set HTTP write trace logging. Thanks @vladlopes.
- [#3027](https://github.com/influxdata/influxdb/pull/3027): Enforce minimum retention policy duration of 1 hour.
- [#3030](https://github.com/influxdata/influxdb/pull/3030): Fix excessive logging of shard creation.
- [#3038](https://github.com/influxdata/influxdb/pull/3038): Don't check deleted shards for precreation. Thanks @vladlopes.
- [#3033](https://github.com/influxdata/influxdb/pull/3033): Add support for marshaling `uint64` in client.
- [#3090](https://github.com/influxdata/influxdb/pull/3090): Remove database from TSDB index on DROP DATABASE.
- [#2944](https://github.com/influxdata/influxdb/issues/2944): Don't require "WHERE time" when creating continuous queries.
- [#3075](https://github.com/influxdata/influxdb/pull/3075): GROUP BY correctly when different tags have same value.
- [#3078](https://github.com/influxdata/influxdb/pull/3078): Fix CLI panic on malformed INSERT.
- [#2102](https://github.com/influxdata/influxdb/issues/2102): Re-work Graphite input and metric processing
- [#2996](https://github.com/influxdata/influxdb/issues/2996): Graphite Input Parsing
- [#3136](https://github.com/influxdata/influxdb/pull/3136): Fix various issues with init.d script. Thanks @ miguelcnf.
- [#2996](https://github.com/influxdata/influxdb/issues/2996): Graphite Input Parsing
- [#3127](https://github.com/influxdata/influxdb/issues/3127): Trying to insert a number larger than the largest signed 64-bit number kills influxd
- [#3131](https://github.com/influxdata/influxdb/pull/3131): Copy batch tags to each point before marshalling
- [#3155](https://github.com/influxdata/influxdb/pull/3155): Instantiate UDP batcher before listening for UDP traffic, otherwise a panic may result.
- [#2678](https://github.com/influxdata/influxdb/issues/2678): Server allows tags with an empty string for the key and/or value
- [#3061](https://github.com/influxdata/influxdb/issues/3061): syntactically incorrect line protocol insert panics the database
- [#2608](https://github.com/influxdata/influxdb/issues/2608): drop measurement while writing points to that measurement has race condition that can panic
- [#3183](https://github.com/influxdata/influxdb/issues/3183): using line protocol measurement names cannot contain commas
- [#3193](https://github.com/influxdata/influxdb/pull/3193): Fix panic for SHOW STATS and in collectd
- [#3102](https://github.com/influxdata/influxdb/issues/3102): Add authentication cache
- [#3209](https://github.com/influxdata/influxdb/pull/3209): Dump Run() errors to stderr
- [#3217](https://github.com/influxdata/influxdb/pull/3217): Allow WAL partition flush delay to be configurable.

## v0.9.0 [2015-06-11]

### Bugfixes

- [#2869](https://github.com/influxdata/influxdb/issues/2869): Adding field to existing measurement causes panic
- [#2849](https://github.com/influxdata/influxdb/issues/2849): RC32: Frequent write errors
- [#2700](https://github.com/influxdata/influxdb/issues/2700): Incorrect error message in database EncodeFields
- [#2897](https://github.com/influxdata/influxdb/pull/2897): Ensure target Graphite database exists
- [#2898](https://github.com/influxdata/influxdb/pull/2898): Ensure target openTSDB database exists
- [#2895](https://github.com/influxdata/influxdb/pull/2895): Use Graphite input defaults where necessary
- [#2900](https://github.com/influxdata/influxdb/pull/2900): Use openTSDB input defaults where necessary
- [#2886](https://github.com/influxdata/influxdb/issues/2886): Refactor backup & restore
- [#2804](https://github.com/influxdata/influxdb/pull/2804): BREAKING: change time literals to be single quoted in InfluxQL. Thanks @nvcook42!
- [#2906](https://github.com/influxdata/influxdb/pull/2906): Restrict replication factor to the cluster size
- [#2905](https://github.com/influxdata/influxdb/pull/2905): Restrict clusters to 3 peers
- [#2904](https://github.com/influxdata/influxdb/pull/2904): Re-enable server reporting.
- [#2917](https://github.com/influxdata/influxdb/pull/2917): Fix int64 field values.
- [#2920](https://github.com/influxdata/influxdb/issues/2920): Ensure collectd database exists

## v0.9.0-rc33 [2015-06-09]

### Bugfixes

- [#2816](https://github.com/influxdata/influxdb/pull/2816): Enable UDP service. Thanks @renan-
- [#2824](https://github.com/influxdata/influxdb/pull/2824): Add missing call to WaitGroup.Done in execConn. Thanks @liyichao
- [#2823](https://github.com/influxdata/influxdb/pull/2823): Convert OpenTSDB to a service.
- [#2838](https://github.com/influxdata/influxdb/pull/2838): Set auto-created retention policy period to infinite.
- [#2829](https://github.com/influxdata/influxdb/pull/2829): Re-enable Graphite support as a new Service-style component.
- [#2814](https://github.com/influxdata/influxdb/issues/2814): Convert collectd to a service.
- [#2852](https://github.com/influxdata/influxdb/pull/2852): Don't panic when altering retention policies. Thanks for the report @huhongbo
- [#2857](https://github.com/influxdata/influxdb/issues/2857): Fix parsing commas in string field values.
- [#2833](https://github.com/influxdata/influxdb/pull/2833): Make the default config valid.
- [#2859](https://github.com/influxdata/influxdb/pull/2859): Fix panic on aggregate functions.
- [#2878](https://github.com/influxdata/influxdb/pull/2878): Re-enable shard precreation.
- [2865](https://github.com/influxdata/influxdb/pull/2865) -- Return an empty set of results if database does not exist in shard metadata.

### Features
- [2858](https://github.com/influxdata/influxdb/pull/2858): Support setting openTSDB write consistency.

## v0.9.0-rc32 [2015-06-07]

### Release Notes

This released introduced an updated write path and clustering design. The data format has also changed, so you'll need to wipe out your data to upgrade from RC31. There should be no other data changes before v0.9.0 is released.

### Features
- [#1997](https://github.com/influxdata/influxdb/pull/1997): Update SELECT * to return tag values.
- [#2599](https://github.com/influxdata/influxdb/issues/2599): Add "epoch" URL param and return JSON time values as epoch instead of date strings.
- [#2682](https://github.com/influxdata/influxdb/issues/2682): Adding pr checklist to CONTRIBUTING.md
- [#2683](https://github.com/influxdata/influxdb/issues/2683): Add batching support to Graphite inputs.
- [#2687](https://github.com/influxdata/influxdb/issues/2687): Add batching support to Collectd inputs.
- [#2696](https://github.com/influxdata/influxdb/pull/2696): Add line protocol. This is now the preferred way to write data.
- [#2751](https://github.com/influxdata/influxdb/pull/2751): Add UDP input. UDP only supports the line protocol now.
- [#2684](https://github.com/influxdata/influxdb/pull/2684): Include client timeout configuration. Thanks @vladlopes!

### Bugfixes
- [#2776](https://github.com/influxdata/influxdb/issues/2776): Re-implement retention policy enforcement.
- [#2635](https://github.com/influxdata/influxdb/issues/2635): Fix querying against boolean field in WHERE clause.
- [#2644](https://github.com/influxdata/influxdb/issues/2644): Make SHOW queries work with FROM /<regex>/.
- [#2501](https://github.com/influxdata/influxdb/issues/2501): Name the FlagSet for the shell and add a version flag. Thanks @neonstalwart
- [#2647](https://github.com/influxdata/influxdb/issues/2647): Fixes typos in sample config file - thanks @claws!

## v0.9.0-rc31 [2015-05-21]

### Features
- [#1822](https://github.com/influxdata/influxdb/issues/1822): Wire up DERIVATIVE aggregate
- [#1477](https://github.com/influxdata/influxdb/issues/1477): Wire up non_negative_derivative function
- [#2557](https://github.com/influxdata/influxdb/issues/2557): Fix false positive error with `GROUP BY time`
- [#1891](https://github.com/influxdata/influxdb/issues/1891): Wire up COUNT DISTINCT aggregate
- [#1989](https://github.com/influxdata/influxdb/issues/1989): Implement `SELECT tagName FROM m`

### Bugfixes
- [#2545](https://github.com/influxdata/influxdb/pull/2545): Use "value" as the field name for graphite input. Thanks @cannium.
- [#2558](https://github.com/influxdata/influxdb/pull/2558): Fix client response check - thanks @vladlopes!
- [#2566](https://github.com/influxdata/influxdb/pull/2566): Wait until each data write has been commited by the Raft cluster.
- [#2602](https://github.com/influxdata/influxdb/pull/2602): CLI execute command exits without cleaning up liner package.
- [#2610](https://github.com/influxdata/influxdb/pull/2610): Fix shard group creation
- [#2596](https://github.com/influxdata/influxdb/pull/2596): RC30: `panic: runtime error: index out of range` when insert data points.
- [#2592](https://github.com/influxdata/influxdb/pull/2592): Should return an error if user attempts to group by a field.
- [#2499](https://github.com/influxdata/influxdb/pull/2499): Issuing a select query with tag as a values causes panic.
- [#2612](https://github.com/influxdata/influxdb/pull/2612): Query planner should validate distinct is passed a field.
- [#2531](https://github.com/influxdata/influxdb/issues/2531): Fix select with 3 or more terms in where clause.
- [#2564](https://github.com/influxdata/influxdb/issues/2564): Change "name" to "measurement" in JSON for writes.

## PRs
- [#2569](https://github.com/influxdata/influxdb/pull/2569): Add derivative functions
- [#2598](https://github.com/influxdata/influxdb/pull/2598): Implement tag support in SELECT statements
- [#2624](https://github.com/influxdata/influxdb/pull/2624): Remove references to SeriesID in `DROP SERIES` handlers.

## v0.9.0-rc30 [2015-05-12]

### Release Notes

This release has a breaking API change for writes -- the field previously called `timestamp` has been renamed to `time`.

### Features
- [#2254](https://github.com/influxdata/influxdb/pull/2254): Add Support for OpenTSDB HTTP interface. Thanks @tcolgate
- [#2525](https://github.com/influxdata/influxdb/pull/2525): Serve broker diagnostics over HTTP
- [#2186](https://github.com/influxdata/influxdb/pull/2186): The default status code for queries is now `200 OK`
- [#2298](https://github.com/influxdata/influxdb/pull/2298): Successful writes now return a status code of `204 No Content` - thanks @neonstalwart!
- [#2549](https://github.com/influxdata/influxdb/pull/2549): Raft election timeout to 5 seconds, so system is more forgiving of CPU loads.
- [#2568](https://github.com/influxdata/influxdb/pull/2568): Wire up SELECT DISTINCT.

### Bugfixes
- [#2535](https://github.com/influxdata/influxdb/pull/2535): Return exit status 0 if influxd already running. Thanks @haim0n.
- [#2521](https://github.com/influxdata/influxdb/pull/2521): Don't truncate topic data until fully replicated.
- [#2509](https://github.com/influxdata/influxdb/pull/2509): Parse config file correctly during restore. Thanks @neonstalwart
- [#2536](https://github.com/influxdata/influxdb/issues/2532): Set leader ID on restart of single-node cluster.
- [#2448](https://github.com/influxdata/influxdb/pull/2448): Fix inconsistent data type - thanks @cannium!
- [#2108](https://github.com/influxdata/influxdb/issues/2108): Change `timestamp` to `time` - thanks @neonstalwart!
- [#2539](https://github.com/influxdata/influxdb/issues/2539): Add additional vote request logging.
- [#2541](https://github.com/influxdata/influxdb/issues/2541): Update messaging client connection index with every message.
- [#2542](https://github.com/influxdata/influxdb/issues/2542): Throw parser error for invalid aggregate without where time.
- [#2548](https://github.com/influxdata/influxdb/issues/2548): Return an error when numeric aggregate applied to non-numeric data.
- [#2487](https://github.com/influxdata/influxdb/issues/2487): Aggregate query with exact timestamp causes panic. Thanks @neonstalwart!
- [#2552](https://github.com/influxdata/influxdb/issues/2552): Run CQ that is actually passed into go-routine.
- [#2553](https://github.com/influxdata/influxdb/issues/2553): Fix race condition during CQ execution.
- [#2557](https://github.com/influxdata/influxdb/issues/2557): RC30 WHERE time filter Regression.

## v0.9.0-rc29 [2015-05-05]

### Features
- [#2410](https://github.com/influxdata/influxdb/pull/2410): If needed, brokers respond with data nodes for peer shard replication.
- [#2469](https://github.com/influxdata/influxdb/pull/2469): Reduce default max topic size from 1GB to 50MB.
- [#1824](https://github.com/influxdata/influxdb/pull/1824): Wire up MEDIAN aggregate. Thanks @neonstalwart!

### Bugfixes
- [#2446](https://github.com/influxdata/influxdb/pull/2446): Correctly count number of queries executed. Thanks @neonstalwart
- [#2452](https://github.com/influxdata/influxdb/issues/2452): Fix panic with shard stats on multiple clusters
- [#2453](https://github.com/influxdata/influxdb/pull/2453): Do not require snapshot on Log.WriteEntriesTo().
- [#2460](https://github.com/influxdata/influxdb/issues/2460): Collectd input should use "value" for fields values. Fixes 2412. Thanks @josh-padnick
- [#2465](https://github.com/influxdata/influxdb/pull/2465): HTTP response logging paniced with chunked requests. Thanks @Jackkoz
- [#2475](https://github.com/influxdata/influxdb/pull/2475): RLock server when checking if shards groups are required during write.
- [#2471](https://github.com/influxdata/influxdb/issues/2471): Function calls normalized to be lower case. Fixes percentile not working when called uppercase. Thanks @neonstalwart
- [#2281](https://github.com/influxdata/influxdb/issues/2281): Fix Bad Escape error when parsing regex

## v0.9.0-rc28 [2015-04-27]

### Features
- [#2410](https://github.com/influxdata/influxdb/pull/2410) Allow configuration of Raft timers
- [#2354](https://github.com/influxdata/influxdb/pull/2354) Wire up STDDEV. Thanks @neonstalwart!

### Bugfixes
- [#2374](https://github.com/influxdata/influxdb/issues/2374): Two different panics during SELECT percentile
- [#2404](https://github.com/influxdata/influxdb/pull/2404): Mean and percentile function fixes
- [#2408](https://github.com/influxdata/influxdb/pull/2408): Fix snapshot 500 error
- [#1896](https://github.com/influxdata/influxdb/issues/1896): Excessive heartbeater logging of "connection refused" on cluster node stop
- [#2418](https://github.com/influxdata/influxdb/pull/2418): Fix raft node getting stuck in candidate state
- [#2415](https://github.com/influxdata/influxdb/pull/2415): Raft leader ID now set on election after failover. Thanks @xiaost
- [#2426](https://github.com/influxdata/influxdb/pull/2426): Fix race condition around listener address in openTSDB server.
- [#2426](https://github.com/influxdata/influxdb/pull/2426): Fix race condition around listener address in Graphite server.
- [#2429](https://github.com/influxdata/influxdb/pull/2429): Ensure no field value is null.
- [#2431](https://github.com/influxdata/influxdb/pull/2431): Always append shard path in diags. Thanks @marcosnils
- [#2441](https://github.com/influxdata/influxdb/pull/2441): Correctly release server RLock during "drop series".
- [#2445](https://github.com/influxdata/influxdb/pull/2445): Read locks and data race fixes

## v0.9.0-rc27 [04-23-2015]

### Features
- [#2398](https://github.com/influxdata/influxdb/pull/2398) Track more stats and report errors for shards.

### Bugfixes
- [#2370](https://github.com/influxdata/influxdb/pull/2370): Fix data race in openTSDB endpoint.
- [#2371](https://github.com/influxdata/influxdb/pull/2371): Don't set client to nil when closing broker Fixes #2352
- [#2372](https://github.com/influxdata/influxdb/pull/2372): Fix data race in graphite endpoint.
- [#2373](https://github.com/influxdata/influxdb/pull/2373): Actually allow HTTP logging to be controlled.
- [#2376](https://github.com/influxdata/influxdb/pull/2376): Encode all types of integers. Thanks @jtakkala.
- [#2376](https://github.com/influxdata/influxdb/pull/2376): Add shard path to existing diags value. Fix issue #2369.
- [#2386](https://github.com/influxdata/influxdb/pull/2386): Fix shard datanodes stats getting appended too many times
- [#2393](https://github.com/influxdata/influxdb/pull/2393): Fix default hostname for connecting to cluster.
- [#2390](https://github.com/influxdata/influxdb/pull/2390): Handle large sums when calculating means - thanks @neonstalwart!
- [#2391](https://github.com/influxdata/influxdb/pull/2391): Unable to write points through Go client when authentication enabled
- [#2400](https://github.com/influxdata/influxdb/pull/2400): Always send auth headers for client requests if present

## v0.9.0-rc26 [04-21-2015]

### Features
- [#2301](https://github.com/influxdata/influxdb/pull/2301): Distributed query load balancing and failover
- [#2336](https://github.com/influxdata/influxdb/pull/2336): Handle distributed queries when shards != data nodes
- [#2353](https://github.com/influxdata/influxdb/pull/2353): Distributed Query/Clustering Fixes

### Bugfixes
- [#2297](https://github.com/influxdata/influxdb/pull/2297): create /var/run during startup. Thanks @neonstalwart.
- [#2312](https://github.com/influxdata/influxdb/pull/2312): Re-use httpclient for continuous queries
- [#2318](https://github.com/influxdata/influxdb/pull/2318): Remove pointless use of 'done' channel for collectd.
- [#2242](https://github.com/influxdata/influxdb/pull/2242): Distributed Query should balance requests
- [#2243](https://github.com/influxdata/influxdb/pull/2243): Use Limit Reader instead of fixed 1MB/1GB slice for DQ
- [#2190](https://github.com/influxdata/influxdb/pull/2190): Implement failover to other data nodes for distributed queries
- [#2324](https://github.com/influxdata/influxdb/issues/2324): Race in Broker.Close()/Broker.RunContinousQueryProcessing()
- [#2325](https://github.com/influxdata/influxdb/pull/2325): Cluster open fixes
- [#2326](https://github.com/influxdata/influxdb/pull/2326): Fix parse error in CREATE CONTINUOUS QUERY
- [#2300](https://github.com/influxdata/influxdb/pull/2300): Refactor integration tests.  Properly close Graphite/OpenTSDB listeners.
- [#2338](https://github.com/influxdata/influxdb/pull/2338): Fix panic if tag key isn't double quoted when it should have been
- [#2340](https://github.com/influxdata/influxdb/pull/2340): Fix SHOW DIAGNOSTICS panic if any shard was non-local.
- [#2351](https://github.com/influxdata/influxdb/pull/2351): Fix data race by rlocking shard during diagnostics.
- [#2348](https://github.com/influxdata/influxdb/pull/2348): Data node fail to join cluster in 0.9.0rc25
- [#2343](https://github.com/influxdata/influxdb/pull/2343): Node falls behind Metastore updates
- [#2334](https://github.com/influxdata/influxdb/pull/2334): Test Partial replication is very problematic
- [#2272](https://github.com/influxdata/influxdb/pull/2272): clustering: influxdb 0.9.0-rc23 panics when doing a GET with merge_metrics in a
- [#2350](https://github.com/influxdata/influxdb/pull/2350): Issue fix for :influxd -hostname localhost.
- [#2367](https://github.com/influxdata/influxdb/pull/2367): PR for issue #2350 - Always use localhost, not host name.

## v0.9.0-rc25 [2015-04-15]

### Bugfixes
- [#2282](https://github.com/influxdata/influxdb/pull/2282): Use "value" as field name for OpenTSDB input.
- [#2283](https://github.com/influxdata/influxdb/pull/2283): Fix bug when restarting an entire existing cluster.
- [#2293](https://github.com/influxdata/influxdb/pull/2293): Open cluster listener before starting broker.
- [#2287](https://github.com/influxdata/influxdb/pull/2287): Fix data race during SHOW RETENTION POLICIES.
- [#2288](https://github.com/influxdata/influxdb/pull/2288): Fix expression parsing bug.
- [#2294](https://github.com/influxdata/influxdb/pull/2294): Fix async response flushing (invalid chunked response error).

## Features
- [#2276](https://github.com/influxdata/influxdb/pull/2276): Broker topic truncation.
- [#2292](https://github.com/influxdata/influxdb/pull/2292): Wire up drop CQ statement - thanks @neonstalwart!
- [#2290](https://github.com/influxdata/influxdb/pull/2290): Allow hostname argument to override default config - thanks @neonstalwart!
- [#2295](https://github.com/influxdata/influxdb/pull/2295): Use nil as default return value for MapCount - thanks @neonstalwart!
- [#2246](https://github.com/influxdata/influxdb/pull/2246): Allow HTTP logging to be controlled.

## v0.9.0-rc24 [2015-04-13]

### Bugfixes
- [#2255](https://github.com/influxdata/influxdb/pull/2255): Fix panic when changing default retention policy.
- [#2257](https://github.com/influxdata/influxdb/pull/2257): Add "snapshotting" pseudo state & log entry cache.
- [#2261](https://github.com/influxdata/influxdb/pull/2261): Support int64 value types.
- [#2191](https://github.com/influxdata/influxdb/pull/2191): Case-insensitive check for "fill"
- [#2274](https://github.com/influxdata/influxdb/pull/2274): Snapshot and HTTP API endpoints
- [#2265](https://github.com/influxdata/influxdb/pull/2265): Fix auth for CLI.

## v0.9.0-rc23 [2015-04-11]

### Features
- [#2202](https://github.com/influxdata/influxdb/pull/2202): Initial implementation of Distributed Queries
- [#2202](https://github.com/influxdata/influxdb/pull/2202): 64-bit Series IDs. INCOMPATIBLE WITH PREVIOUS DATASTORES.

### Bugfixes
- [#2225](https://github.com/influxdata/influxdb/pull/2225): Make keywords completely case insensitive
- [#2228](https://github.com/influxdata/influxdb/pull/2228): Accept keyword default unquoted in ALTER RETENTION POLICY statement
- [#2236](https://github.com/influxdata/influxdb/pull/2236): Immediate term changes, fix stale write issue, net/http/pprof
- [#2213](https://github.com/influxdata/influxdb/pull/2213): Seed random number generator for election timeout. Thanks @cannium.

## v0.9.0-rc22 [2015-04-09]

### Features
- [#2214](https://github.com/influxdata/influxdb/pull/2214): Added the option to influx CLI to execute single command and exit. Thanks @n1tr0g

### Bugfixes
- [#2223](https://github.com/influxdata/influxdb/pull/2223): Always notify term change on RequestVote

## v0.9.0-rc21 [2015-04-09]

### Features
- [#870](https://github.com/influxdata/influxdb/pull/870): Add support for OpenTSDB telnet input protocol. Thanks @tcolgate
- [#2180](https://github.com/influxdata/influxdb/pull/2180): Allow http write handler to decode gzipped body
- [#2175](https://github.com/influxdata/influxdb/pull/2175): Separate broker and data nodes
- [#2158](https://github.com/influxdata/influxdb/pull/2158): Allow user password to be changed. Thanks @n1tr0g
- [#2201](https://github.com/influxdata/influxdb/pull/2201): Bring back config join URLs
- [#2121](https://github.com/influxdata/influxdb/pull/2121): Parser refactor

### Bugfixes
- [#2181](https://github.com/influxdata/influxdb/pull/2181): Fix panic on "SHOW DIAGNOSTICS".
- [#2170](https://github.com/influxdata/influxdb/pull/2170): Make sure queries on missing tags return 200 status.
- [#2197](https://github.com/influxdata/influxdb/pull/2197): Lock server during Open().
- [#2200](https://github.com/influxdata/influxdb/pull/2200): Re-enable Continuous Queries.
- [#2203](https://github.com/influxdata/influxdb/pull/2203): Fix race condition on continuous queries.
- [#2217](https://github.com/influxdata/influxdb/pull/2217): Only revert to follower if new term is greater.
- [#2219](https://github.com/influxdata/influxdb/pull/2219): Persist term change to disk when candidate. Thanks @cannium

## v0.9.0-rc20 [2015-04-04]

### Features
- [#2128](https://github.com/influxdata/influxdb/pull/2128): Data node discovery from brokers
- [#2142](https://github.com/influxdata/influxdb/pull/2142): Support chunked queries
- [#2154](https://github.com/influxdata/influxdb/pull/2154): Node redirection
- [#2168](https://github.com/influxdata/influxdb/pull/2168): Return raft term from vote, add term logging

### Bugfixes
- [#2147](https://github.com/influxdata/influxdb/pull/2147): Set Go Max procs in a better location
- [#2137](https://github.com/influxdata/influxdb/pull/2137): Refactor `results` to `response`. Breaking Go Client change.
- [#2151](https://github.com/influxdata/influxdb/pull/2151): Ignore replay commands on the metastore.
- [#2152](https://github.com/influxdata/influxdb/issues/2152): Influxd process with stats enabled crashing with 'Unsuported protocol scheme for ""'
- [#2156](https://github.com/influxdata/influxdb/pull/2156): Propagate error when resolving UDP address in Graphite UDP server.
- [#2163](https://github.com/influxdata/influxdb/pull/2163): Fix up paths for default data and run storage.
- [#2164](https://github.com/influxdata/influxdb/pull/2164): Append STDOUT/STDERR in initscript.
- [#2165](https://github.com/influxdata/influxdb/pull/2165): Better name for config section for stats and diags.
- [#2165](https://github.com/influxdata/influxdb/pull/2165): Monitoring database and retention policy are not configurable.
- [#2167](https://github.com/influxdata/influxdb/pull/2167): Add broker log recovery.
- [#2166](https://github.com/influxdata/influxdb/pull/2166): Don't panic if presented with a field of unknown type.
- [#2149](https://github.com/influxdata/influxdb/pull/2149): Fix unit tests for win32 when directory doesn't exist.
- [#2150](https://github.com/influxdata/influxdb/pull/2150): Fix unit tests for win32 when a connection is refused.

## v0.9.0-rc19 [2015-04-01]

### Features
- [#2143](https://github.com/influxdata/influxdb/pull/2143): Add raft term logging.

### Bugfixes
- [#2145](https://github.com/influxdata/influxdb/pull/2145): Encode toml durations correctly which fixes default configuration generation `influxd config`.

## v0.9.0-rc18 [2015-03-31]

### Bugfixes
- [#2100](https://github.com/influxdata/influxdb/pull/2100): Use channel to synchronize collectd shutdown.
- [#2100](https://github.com/influxdata/influxdb/pull/2100): Synchronize access to shard index.
- [#2131](https://github.com/influxdata/influxdb/pull/2131): Optimize marshalTags().
- [#2130](https://github.com/influxdata/influxdb/pull/2130): Make fewer calls to marshalTags().
- [#2105](https://github.com/influxdata/influxdb/pull/2105): Support != for tag values. Fix issue #2097, thanks to @smonkewitz for bug report.
- [#2105](https://github.com/influxdata/influxdb/pull/2105): Support !~ tags values.
- [#2138](https://github.com/influxdata/influxdb/pull/2136): Use map for marshaledTags cache.

## v0.9.0-rc17 [2015-03-29]

### Features
- [#2076](https://github.com/influxdata/influxdb/pull/2076): Separate stdout and stderr output in init.d script
- [#2091](https://github.com/influxdata/influxdb/pull/2091): Support disabling snapshot endpoint.
- [#2081](https://github.com/influxdata/influxdb/pull/2081): Support writing diagnostic data into the internal database.
- [#2095](https://github.com/influxdata/influxdb/pull/2095): Improved InfluxDB client docs. Thanks @derailed

### Bugfixes
- [#2093](https://github.com/influxdata/influxdb/pull/2093): Point precision not marshalled correctly. Thanks @derailed
- [#2084](https://github.com/influxdata/influxdb/pull/2084): Allowing leading underscores in identifiers.
- [#2080](https://github.com/influxdata/influxdb/pull/2080): Graphite logs in seconds, not milliseconds.
- [#2101](https://github.com/influxdata/influxdb/pull/2101): SHOW DATABASES should name returned series "databases".
- [#2104](https://github.com/influxdata/influxdb/pull/2104): Include NEQ when calculating field filters.
- [#2112](https://github.com/influxdata/influxdb/pull/2112): Set GOMAXPROCS on startup. This may have been causing extra leader elections, which would cause a number of other bugs or instability.
- [#2111](https://github.com/influxdata/influxdb/pull/2111) and [#2025](https://github.com/influxdata/influxdb/issues/2025): Raft stability fixes. Non-contiguous log error and others.
- [#2114](https://github.com/influxdata/influxdb/pull/2114): Correctly start influxd on platforms without start-stop-daemon.

## v0.9.0-rc16 [2015-03-24]

### Features
- [#2058](https://github.com/influxdata/influxdb/pull/2058): Track number of queries executed in stats.
- [#2059](https://github.com/influxdata/influxdb/pull/2059): Retention policies sorted by name on return to client.
- [#2061](https://github.com/influxdata/influxdb/pull/2061): Implement SHOW DIAGNOSTICS.
- [#2064](https://github.com/influxdata/influxdb/pull/2064): Allow init.d script to return influxd version.
- [#2053](https://github.com/influxdata/influxdb/pull/2053): Implment backup and restore.
- [#1631](https://github.com/influxdata/influxdb/pull/1631): Wire up DROP CONTINUOUS QUERY.

### Bugfixes
- [#2037](https://github.com/influxdata/influxdb/pull/2037): Don't check 'configExists' at Run() level.
- [#2039](https://github.com/influxdata/influxdb/pull/2039): Don't panic if getting current user fails.
- [#2034](https://github.com/influxdata/influxdb/pull/2034): GROUP BY should require an aggregate.
- [#2040](https://github.com/influxdata/influxdb/pull/2040): Add missing top-level help for config command.
- [#2057](https://github.com/influxdata/influxdb/pull/2057): Move racy "in order" test to integration test suite.
- [#2060](https://github.com/influxdata/influxdb/pull/2060): Reload server shard map on restart.
- [#2068](https://github.com/influxdata/influxdb/pull/2068): Fix misspelled JSON field.
- [#2067](https://github.com/influxdata/influxdb/pull/2067): Fixed issue where some queries didn't properly pull back data (introduced in RC15). Fixing intervals for GROUP BY.

## v0.9.0-rc15 [2015-03-19]

### Features
- [#2000](https://github.com/influxdata/influxdb/pull/2000): Log broker path when broker fails to start. Thanks @gst.
- [#2007](https://github.com/influxdata/influxdb/pull/2007): Track shard-level stats.

### Bugfixes
- [#2001](https://github.com/influxdata/influxdb/pull/2001): Ensure measurement not found returns status code 200.
- [#1985](https://github.com/influxdata/influxdb/pull/1985): Set content-type JSON header before actually writing header. Thanks @dstrek.
- [#2003](https://github.com/influxdata/influxdb/pull/2003): Set timestamp when writing monitoring stats.
- [#2004](https://github.com/influxdata/influxdb/pull/2004): Limit group by to MaxGroupByPoints (currently 100,000).
- [#2016](https://github.com/influxdata/influxdb/pull/2016): Fixing bucket alignment for group by. Thanks @jnutzmann
- [#2021](https://github.com/influxdata/influxdb/pull/2021): Remove unnecessary formatting from log message. Thanks @simonkern


## v0.9.0-rc14 [2015-03-18]

### Bugfixes
- [#1999](https://github.com/influxdata/influxdb/pull/1999): Return status code 200 for measurement not found errors on show series.

## v0.9.0-rc13 [2015-03-17]

### Features
- [#1974](https://github.com/influxdata/influxdb/pull/1974): Add time taken for request to the http server logs.

### Bugfixes
- [#1971](https://github.com/influxdata/influxdb/pull/1971): Fix leader id initialization.
- [#1975](https://github.com/influxdata/influxdb/pull/1975): Require `q` parameter for query endpoint.
- [#1969](https://github.com/influxdata/influxdb/pull/1969): Print loaded config.
- [#1987](https://github.com/influxdata/influxdb/pull/1987): Fix config print startup statement for when no config is provided.
- [#1990](https://github.com/influxdata/influxdb/pull/1990): Drop measurement was taking too long due to transactions.

## v0.9.0-rc12 [2015-03-15]

### Bugfixes
- [#1942](https://github.com/influxdata/influxdb/pull/1942): Sort wildcard names.
- [#1957](https://github.com/influxdata/influxdb/pull/1957): Graphite numbers are always float64.
- [#1955](https://github.com/influxdata/influxdb/pull/1955): Prohibit creation of databases with no name. Thanks @dullgiulio
- [#1952](https://github.com/influxdata/influxdb/pull/1952): Handle delete statement with an error. Thanks again to @dullgiulio

### Features
- [#1935](https://github.com/influxdata/influxdb/pull/1935): Implement stateless broker for Raft.
- [#1936](https://github.com/influxdata/influxdb/pull/1936): Implement "SHOW STATS" and self-monitoring

### Features
- [#1909](https://github.com/influxdata/influxdb/pull/1909): Implement a dump command.

## v0.9.0-rc11 [2015-03-13]

### Bugfixes
- [#1917](https://github.com/influxdata/influxdb/pull/1902): Creating Infinite Retention Policy Failed.
- [#1758](https://github.com/influxdata/influxdb/pull/1758): Add Graphite Integration Test.
- [#1929](https://github.com/influxdata/influxdb/pull/1929): Default Retention Policy incorrectly auto created.
- [#1930](https://github.com/influxdata/influxdb/pull/1930): Auto create database for graphite if not specified.
- [#1908](https://github.com/influxdata/influxdb/pull/1908): Cosmetic CLI output fixes.
- [#1931](https://github.com/influxdata/influxdb/pull/1931): Add default column to SHOW RETENTION POLICIES.
- [#1937](https://github.com/influxdata/influxdb/pull/1937): OFFSET should be allowed to be 0.

### Features
- [#1902](https://github.com/influxdata/influxdb/pull/1902): Enforce retention policies to have a minimum duration.
- [#1906](https://github.com/influxdata/influxdb/pull/1906): Add show servers to query language.
- [#1925](https://github.com/influxdata/influxdb/pull/1925): Add `fill(none)`, `fill(previous)`, and `fill(<num>)` to queries.

## v0.9.0-rc10 [2015-03-09]

### Bugfixes
- [#1867](https://github.com/influxdata/influxdb/pull/1867): Fix race accessing topic replicas map
- [#1864](https://github.com/influxdata/influxdb/pull/1864): fix race in startStateLoop
- [#1753](https://github.com/influxdata/influxdb/pull/1874): Do Not Panic on Missing Dirs
- [#1877](https://github.com/influxdata/influxdb/pull/1877): Broker clients track broker leader
- [#1862](https://github.com/influxdata/influxdb/pull/1862): Fix memory leak in `httpd.serveWait`. Thanks @mountkin
- [#1883](https://github.com/influxdata/influxdb/pull/1883): RLock server during retention policy enforcement. Thanks @grisha
- [#1868](https://github.com/influxdata/influxdb/pull/1868): Use `BatchPoints` for `client.Write` method. Thanks @vladlopes, @georgmu, @d2g, @evanphx, @akolosov.
- [#1881](https://github.com/influxdata/influxdb/pull/1881): Update documentation for `client` package.  Misc library tweaks.
- Fix queries with multiple where clauses on tags, times and fields. Fix queries that have where clauses on fields not in the select

### Features
- [#1875](https://github.com/influxdata/influxdb/pull/1875): Support trace logging of Raft.
- [#1895](https://github.com/influxdata/influxdb/pull/1895): Auto-create a retention policy when a database is created.
- [#1897](https://github.com/influxdata/influxdb/pull/1897): Pre-create shard groups.
- [#1900](https://github.com/influxdata/influxdb/pull/1900): Change `LIMIT` to `SLIMIT` and implement `LIMIT` and `OFFSET`

## v0.9.0-rc9 [2015-03-06]

### Bugfixes
- [#1872](https://github.com/influxdata/influxdb/pull/1872): Fix "stale term" errors with raft

## v0.9.0-rc8 [2015-03-05]

### Bugfixes
- [#1836](https://github.com/influxdata/influxdb/pull/1836): Store each parsed shell command in history file.
- [#1789](https://github.com/influxdata/influxdb/pull/1789): add --config-files option to fpm command. Thanks @kylezh
- [#1859](https://github.com/influxdata/influxdb/pull/1859): Queries with a `GROUP BY *` clause were returning a 500 if done against a measurement that didn't exist

### Features
- [#1755](https://github.com/influxdata/influxdb/pull/1848): Support JSON data ingest over UDP
- [#1857](https://github.com/influxdata/influxdb/pull/1857): Support retention policies with infinite duration
- [#1858](https://github.com/influxdata/influxdb/pull/1858): Enable detailed tracing of write path

## v0.9.0-rc7 [2015-03-02]

### Features
- [#1813](https://github.com/influxdata/influxdb/pull/1813): Queries for missing measurements or fields now return a 200 with an error message in the series JSON.
- [#1826](https://github.com/influxdata/influxdb/pull/1826), [#1827](https://github.com/influxdata/influxdb/pull/1827): Fixed queries with `WHERE` clauses against fields.

### Bugfixes

- [#1744](https://github.com/influxdata/influxdb/pull/1744): Allow retention policies to be modified without specifying replication factor. Thanks @kylezh
- [#1809](https://github.com/influxdata/influxdb/pull/1809): Packaging post-install script unconditionally removes init.d symlink. Thanks @sineos

## v0.9.0-rc6 [2015-02-27]

### Bugfixes

- [#1780](https://github.com/influxdata/influxdb/pull/1780): Malformed identifiers get through the parser
- [#1775](https://github.com/influxdata/influxdb/pull/1775): Panic "index out of range" on some queries
- [#1744](https://github.com/influxdata/influxdb/pull/1744): Select shard groups which completely encompass time range. Thanks @kylezh.

## v0.9.0-rc5 [2015-02-27]

### Bugfixes

- [#1752](https://github.com/influxdata/influxdb/pull/1752): remove debug log output from collectd.
- [#1720](https://github.com/influxdata/influxdb/pull/1720): Parse Series IDs as unsigned 32-bits.
- [#1767](https://github.com/influxdata/influxdb/pull/1767): Drop Series was failing across shards.  Issue #1761.
- [#1773](https://github.com/influxdata/influxdb/pull/1773): Fix bug when merging series together that have unequal number of points in a group by interval
- [#1771](https://github.com/influxdata/influxdb/pull/1771): Make `SHOW SERIES` return IDs and support `LIMIT` and `OFFSET`

### Features

- [#1698](https://github.com/influxdata/influxdb/pull/1698): Wire up DROP MEASUREMENT

## v0.9.0-rc4 [2015-02-24]

### Bugfixes

- Fix authentication issue with continuous queries
- Print version in the log on startup

## v0.9.0-rc3 [2015-02-23]

### Features

- [#1659](https://github.com/influxdata/influxdb/pull/1659): WHERE against regexes: `WHERE =~ '.*asdf'
- [#1580](https://github.com/influxdata/influxdb/pull/1580): Add support for fields with bool, int, or string data types
- [#1687](https://github.com/influxdata/influxdb/pull/1687): Change `Rows` to `Series` in results output. BREAKING API CHANGE
- [#1629](https://github.com/influxdata/influxdb/pull/1629): Add support for `DROP SERIES` queries
- [#1632](https://github.com/influxdata/influxdb/pull/1632): Add support for `GROUP BY *` to return all series within a measurement
- [#1689](https://github.com/influxdata/influxdb/pull/1689): Change `SHOW TAG VALUES WITH KEY="foo"` to use the key name in the result. BREAKING API CHANGE
- [#1699](https://github.com/influxdata/influxdb/pull/1699): Add CPU and memory profiling options to daemon
- [#1672](https://github.com/influxdata/influxdb/pull/1672): Add index tracking to metastore. Makes downed node recovery actually work
- [#1591](https://github.com/influxdata/influxdb/pull/1591): Add `spread` aggregate function
- [#1576](https://github.com/influxdata/influxdb/pull/1576): Add `first` and `last` aggregate functions
- [#1573](https://github.com/influxdata/influxdb/pull/1573): Add `stddev` aggregate function
- [#1565](https://github.com/influxdata/influxdb/pull/1565): Add the admin interface back into the server and update for new API
- [#1562](https://github.com/influxdata/influxdb/pull/1562): Enforce retention policies
- [#1700](https://github.com/influxdata/influxdb/pull/1700): Change `Values` to `Fields` on writes.  BREAKING API CHANGE
- [#1706](https://github.com/influxdata/influxdb/pull/1706): Add support for `LIMIT` and `OFFSET`, which work on the number of series returned in a query. To limit the number of data points use a `WHERE time` clause

### Bugfixes

- [#1636](https://github.com/influxdata/influxdb/issues/1636): Don't store number of fields in raw data. THIS IS A BREAKING DATA CHANGE. YOU MUST START WITH A FRESH DATABASE
- [#1701](https://github.com/influxdata/influxdb/pull/1701), [#1667](https://github.com/influxdata/influxdb/pull/1667), [#1663](https://github.com/influxdata/influxdb/pull/1663), [#1615](https://github.com/influxdata/influxdb/pull/1615): Raft fixes
- [#1644](https://github.com/influxdata/influxdb/pull/1644): Add batching support for significantly improved write performance
- [#1704](https://github.com/influxdata/influxdb/pull/1704): Fix queries that pull back raw data (i.e. ones without aggregate functions)
- [#1718](https://github.com/influxdata/influxdb/pull/1718): Return an error on write if any of the points are don't have at least one field
- [#1806](https://github.com/influxdata/influxdb/pull/1806): Fix regex parsing.  Change regex syntax to use / delimiters.


## v0.9.0-rc1,2 [no public release]

### Features

- Support for tags added
- New queries for showing measurement names, tag keys, and tag values
- Renamed shard spaces to retention policies
- Deprecated matching against regex in favor of explicit writing and querying on retention policies
- Pure Go InfluxQL parser
- Switch to BoltDB as underlying datastore
- BoltDB backed metastore to store schema information
- Updated HTTP API to only have two endpoints `/query` and `/write`
- Added all administrative functions to the query language
- Change cluster architecture to have brokers and data nodes
- Switch to streaming Raft implementation
- In memory inverted index of the tag data
- Pure Go implementation!

## v0.8.6 [2014-11-15]

### Features

- [Issue #973](https://github.com/influxdata/influxdb/issues/973). Support
  joining using a regex or list of time series
- [Issue #1068](https://github.com/influxdata/influxdb/issues/1068). Print
  the processor chain when the query is started

### Bugfixes

- [Issue #584](https://github.com/influxdata/influxdb/issues/584). Don't
  panic if the process died while initializing
- [Issue #663](https://github.com/influxdata/influxdb/issues/663). Make
  sure all sub servies are closed when are stopping InfluxDB
- [Issue #671](https://github.com/influxdata/influxdb/issues/671). Fix
  the Makefile package target for Mac OSX
- [Issue #800](https://github.com/influxdata/influxdb/issues/800). Use
  su instead of sudo in the init script. This fixes the startup problem
  on RHEL 6.
- [Issue #925](https://github.com/influxdata/influxdb/issues/925). Don't
  generate invalid query strings for single point queries
- [Issue #943](https://github.com/influxdata/influxdb/issues/943). Don't
  take two snapshots at the same time
- [Issue #947](https://github.com/influxdata/influxdb/issues/947). Exit
  nicely if the daemon doesn't have permission to write to the log.
- [Issue #959](https://github.com/influxdata/influxdb/issues/959). Stop using
  closed connections in the protobuf client.
- [Issue #978](https://github.com/influxdata/influxdb/issues/978). Check
  for valgrind and mercurial in the configure script
- [Issue #996](https://github.com/influxdata/influxdb/issues/996). Fill should
  fill the time range even if no points exists in the given time range
- [Issue #1008](https://github.com/influxdata/influxdb/issues/1008). Return
  an appropriate exit status code depending on whether the process exits
  due to an error or exits gracefully.
- [Issue #1024](https://github.com/influxdata/influxdb/issues/1024). Hitting
  open files limit causes influxdb to create shards in loop.
- [Issue #1069](https://github.com/influxdata/influxdb/issues/1069). Fix
  deprecated interface endpoint in Admin UI.
- [Issue #1076](https://github.com/influxdata/influxdb/issues/1076). Fix
  the timestamps of data points written by the collectd plugin. (Thanks,
  @renchap for reporting this bug)
- [Issue #1078](https://github.com/influxdata/influxdb/issues/1078). Make sure
  we don't resurrect shard directories for shards that have already expired
- [Issue #1085](https://github.com/influxdata/influxdb/issues/1085). Set
  the connection string of the local raft node
- [Issue #1092](https://github.com/influxdata/influxdb/issues/1093). Set
  the connection string of the local node in the raft snapshot.
- [Issue #1100](https://github.com/influxdata/influxdb/issues/1100). Removing
  a non-existent shard space causes the cluster to panic.
- [Issue #1113](https://github.com/influxdata/influxdb/issues/1113). A nil
  engine.ProcessorChain causes a panic.

## v0.8.5 [2014-10-27]

### Features

- [Issue #1055](https://github.com/influxdata/influxdb/issues/1055). Allow
  graphite and collectd input plugins to have separate binding address

### Bugfixes

- [Issue #1058](https://github.com/influxdata/influxdb/issues/1058). Use
  the query language instead of the continuous query endpoints that
  were removed in 0.8.4
- [Issue #1022](https://github.com/influxdata/influxdb/issues/1022). Return
  an +Inf or NaN instead of panicing when we encounter a divide by zero
- [Issue #821](https://github.com/influxdata/influxdb/issues/821). Don't
  scan through points when we hit the limit
- [Issue #1051](https://github.com/influxdata/influxdb/issues/1051). Fix
  timestamps when the collectd is used and low resolution timestamps
  is set.

## v0.8.4 [2014-10-24]

### Bugfixes

- Remove the continuous query api endpoints since the query language
  has all the features needed to list and delete continuous queries.
- [Issue #778](https://github.com/influxdata/influxdb/issues/778). Selecting
  from a non-existent series should give a better error message indicating
  that the series doesn't exist
- [Issue #988](https://github.com/influxdata/influxdb/issues/988). Check
  the arguments of `top()` and `bottom()`
- [Issue #1021](https://github.com/influxdata/influxdb/issues/1021). Make
  redirecting to standard output and standard error optional instead of
  going to `/dev/null`. This can now be configured by setting `$STDOUT`
  in `/etc/default/influxdb`
- [Issue #985](https://github.com/influxdata/influxdb/issues/985). Make
  sure we drop a shard only when there's no one using it. Otherwise, the
  shard can be closed when another goroutine is writing to it which will
  cause random errors and possibly corruption of the database.

### Features

- [Issue #1047](https://github.com/influxdata/influxdb/issues/1047). Allow
  merge() to take a list of series (as opposed to a regex in #72)

## v0.8.4-rc.1 [2014-10-21]

### Bugfixes

- [Issue #1040](https://github.com/influxdata/influxdb/issues/1040). Revert
  to older raft snapshot if the latest one is corrupted
- [Issue #1004](https://github.com/influxdata/influxdb/issues/1004). Querying
  for data outside of existing shards returns an empty response instead of
  throwing a `Couldn't lookup columns` error
- [Issue #1020](https://github.com/influxdata/influxdb/issues/1020). Change
  init script exit codes to conform to the lsb standards. (Thanks, @spuder)
- [Issue #1011](https://github.com/influxdata/influxdb/issues/1011). Fix
  the tarball for homebrew so that rocksdb is included and the directory
  structure is clean
- [Issue #1007](https://github.com/influxdata/influxdb/issues/1007). Fix
  the content type when an error occurs and the client requests
  compression.
- [Issue #916](https://github.com/influxdata/influxdb/issues/916). Set
  the ulimit in the init script with a way to override the limit
- [Issue #742](https://github.com/influxdata/influxdb/issues/742). Fix
  rocksdb for Mac OSX
- [Issue #387](https://github.com/influxdata/influxdb/issues/387). Aggregations
  with group by time(1w), time(1m) and time(1y) (for week, month and
  year respectively) will cause the start time and end time of the bucket
  to fall on the logical boundaries of the week, month or year.
- [Issue #334](https://github.com/influxdata/influxdb/issues/334). Derivative
  for queries with group by time() and fill(), will take the difference
  between the first value in the bucket and the first value of the next
  bucket.
- [Issue #972](https://github.com/influxdata/influxdb/issues/972). Don't
  assign duplicate server ids

### Features

- [Issue #722](https://github.com/influxdata/influxdb/issues/722). Add
  an install target to the Makefile
- [Issue #1032](https://github.com/influxdata/influxdb/issues/1032). Include
  the admin ui static assets in the binary
- [Issue #1019](https://github.com/influxdata/influxdb/issues/1019). Upgrade
  to rocksdb 3.5.1
- [Issue #992](https://github.com/influxdata/influxdb/issues/992). Add
  an input plugin for collectd. (Thanks, @kimor79)
- [Issue #72](https://github.com/influxdata/influxdb/issues/72). Support merge
  for multiple series using regex syntax

## v0.8.3 [2014-09-24]

### Bugfixes

- [Issue #885](https://github.com/influxdata/influxdb/issues/885). Multiple
  queries separated by semicolons work as expected. Queries are process
  sequentially
- [Issue #652](https://github.com/influxdata/influxdb/issues/652). Return an
  error if an invalid column is used in the where clause
- [Issue #794](https://github.com/influxdata/influxdb/issues/794). Fix case
  insensitive regex matching
- [Issue #853](https://github.com/influxdata/influxdb/issues/853). Move
  cluster config from raft to API.
- [Issue #714](https://github.com/influxdata/influxdb/issues/714). Don't
  panic on invalid boolean operators.
- [Issue #843](https://github.com/influxdata/influxdb/issues/843). Prevent blank database names
- [Issue #780](https://github.com/influxdata/influxdb/issues/780). Fix
  fill() for all aggregators
- [Issue #923](https://github.com/influxdata/influxdb/issues/923). Enclose
  table names in double quotes in the result of GetQueryString()
- [Issue #923](https://github.com/influxdata/influxdb/issues/923). Enclose
  table names in double quotes in the result of GetQueryString()
- [Issue #967](https://github.com/influxdata/influxdb/issues/967). Return an
  error if the storage engine can't be created
- [Issue #954](https://github.com/influxdata/influxdb/issues/954). Don't automatically
  create shards which was causing too many shards to be created when used with
  grafana
- [Issue #939](https://github.com/influxdata/influxdb/issues/939). Aggregation should
  ignore null values and invalid values, e.g. strings with mean().
- [Issue #964](https://github.com/influxdata/influxdb/issues/964). Parse
  big int in queries properly.

## v0.8.2 [2014-09-05]

### Bugfixes

- [Issue #886](https://github.com/influxdata/influxdb/issues/886). Update shard space to not set defaults

- [Issue #867](https://github.com/influxdata/influxdb/issues/867). Add option to return shard space mappings in list series

### Bugfixes

- [Issue #652](https://github.com/influxdata/influxdb/issues/652). Return
  a meaningful error if an invalid column is used in where clause
  after joining multiple series

## v0.8.2 [2014-09-08]

### Features

- Added API endpoint to update shard space definitions

### Bugfixes

- [Issue #886](https://github.com/influxdata/influxdb/issues/886). Shard space regexes reset after restart of InfluxDB

## v0.8.1 [2014-09-03]

- [Issue #896](https://github.com/influxdata/influxdb/issues/896). Allow logging to syslog. Thanks @malthe

### Bugfixes

- [Issue #868](https://github.com/influxdata/influxdb/issues/868). Don't panic when upgrading a snapshot from 0.7.x
- [Issue #887](https://github.com/influxdata/influxdb/issues/887). The first continuous query shouldn't trigger backfill if it had backfill disabled
- [Issue #674](https://github.com/influxdata/influxdb/issues/674). Graceful exit when config file is invalid. (Thanks, @DavidBord)
- [Issue #857](https://github.com/influxdata/influxdb/issues/857). More informative list servers api. (Thanks, @oliveagle)

## v0.8.0 [2014-08-22]

### Features

- [Issue #850](https://github.com/influxdata/influxdb/issues/850). Makes the server listing more informative

### Bugfixes

- [Issue #779](https://github.com/influxdata/influxdb/issues/779). Deleting expired shards isn't thread safe.
- [Issue #860](https://github.com/influxdata/influxdb/issues/860). Load database config should validate shard spaces.
- [Issue #862](https://github.com/influxdata/influxdb/issues/862). Data migrator should have option to set delay time.

## v0.8.0-rc.5 [2014-08-15]

### Features

- [Issue #376](https://github.com/influxdata/influxdb/issues/376). List series should support regex filtering
- [Issue #745](https://github.com/influxdata/influxdb/issues/745). Add continuous queries to the database config
- [Issue #746](https://github.com/influxdata/influxdb/issues/746). Add data migration tool for 0.8.0

### Bugfixes

- [Issue #426](https://github.com/influxdata/influxdb/issues/426). Fill should fill the entire time range that is requested
- [Issue #740](https://github.com/influxdata/influxdb/issues/740). Don't emit non existent fields when joining series with different fields
- [Issue #744](https://github.com/influxdata/influxdb/issues/744). Admin site should have all assets locally
- [Issue #767](https://github.com/influxdata/influxdb/issues/768). Remove shards whenever they expire
- [Issue #781](https://github.com/influxdata/influxdb/issues/781). Don't emit non existent fields when joining series with different fields
- [Issue #791](https://github.com/influxdata/influxdb/issues/791). Move database config loader to be an API endpoint
- [Issue #809](https://github.com/influxdata/influxdb/issues/809). Migration path from 0.7 -> 0.8
- [Issue #811](https://github.com/influxdata/influxdb/issues/811). Gogoprotobuf removed `ErrWrongType`, which is depended on by Raft
- [Issue #820](https://github.com/influxdata/influxdb/issues/820). Query non-local shard with time range to avoid getting back points not in time range
- [Issue #827](https://github.com/influxdata/influxdb/issues/827). Don't leak file descriptors in the WAL
- [Issue #830](https://github.com/influxdata/influxdb/issues/830). List series should return series in lexicographic sorted order
- [Issue #831](https://github.com/influxdata/influxdb/issues/831). Move create shard space to be db specific

## v0.8.0-rc.4 [2014-07-29]

### Bugfixes

- [Issue #774](https://github.com/influxdata/influxdb/issues/774). Don't try to parse "inf" shard retention policy
- [Issue #769](https://github.com/influxdata/influxdb/issues/769). Use retention duration when determining expired shards. (Thanks, @shugo)
- [Issue #736](https://github.com/influxdata/influxdb/issues/736). Only db admins should be able to drop a series
- [Issue #713](https://github.com/influxdata/influxdb/issues/713). Null should be a valid fill value
- [Issue #644](https://github.com/influxdata/influxdb/issues/644). Graphite api should write data in batches to the coordinator
- [Issue #740](https://github.com/influxdata/influxdb/issues/740). Panic when distinct fields are selected from an inner join
- [Issue #781](https://github.com/influxdata/influxdb/issues/781). Panic when distinct fields are added after an inner join

## v0.8.0-rc.3 [2014-07-21]

### Bugfixes

- [Issue #752](https://github.com/influxdata/influxdb/issues/752). `./configure` should use goroot to find gofmt
- [Issue #758](https://github.com/influxdata/influxdb/issues/758). Clarify the reason behind graphite input plugin not starting. (Thanks, @otoolep)
- [Issue #759](https://github.com/influxdata/influxdb/issues/759). Don't revert the regex in the shard space. (Thanks, @shugo)
- [Issue #760](https://github.com/influxdata/influxdb/issues/760). Removing a server should remove it from the shard server ids. (Thanks, @shugo)
- [Issue #772](https://github.com/influxdata/influxdb/issues/772). Add sentinel values to all db. This caused the last key in the db to not be fetched properly.


## v0.8.0-rc.2 [2014-07-15]

- This release is to fix a build error in rc1 which caused rocksdb to not be available
- Bump up the `max-open-files` option to 1000 on all storage engines
- Lower the `write-buffer-size` to 1000

## v0.8.0-rc.1 [2014-07-15]

### Features

- [Issue #643](https://github.com/influxdata/influxdb/issues/643). Support pretty print json. (Thanks, @otoolep)
- [Issue #641](https://github.com/influxdata/influxdb/issues/641). Support multiple storage engines
- [Issue #665](https://github.com/influxdata/influxdb/issues/665). Make build tmp directory configurable in the make file. (Thanks, @dgnorton)
- [Issue #667](https://github.com/influxdata/influxdb/issues/667). Enable compression on all GET requests and when writing data
- [Issue #648](https://github.com/influxdata/influxdb/issues/648). Return permissions when listing db users. (Thanks, @nicolai86)
- [Issue #682](https://github.com/influxdata/influxdb/issues/682). Allow continuous queries to run without backfill (Thanks, @dhammika)
- [Issue #689](https://github.com/influxdata/influxdb/issues/689). **REQUIRES DATA MIGRATION** Move metadata into raft
- [Issue #255](https://github.com/influxdata/influxdb/issues/255). Support millisecond precision using `ms` suffix
- [Issue #95](https://github.com/influxdata/influxdb/issues/95). Drop database should not be synchronous
- [Issue #571](https://github.com/influxdata/influxdb/issues/571). Add support for arbitrary number of shard spaces and retention policies
- Default storage engine changed to RocksDB

### Bugfixes

- [Issue #651](https://github.com/influxdata/influxdb/issues/651). Change permissions of symlink which fix some installation issues. (Thanks, @Dieterbe)
- [Issue #670](https://github.com/influxdata/influxdb/issues/670). Don't warn on missing influxdb user on fresh installs
- [Issue #676](https://github.com/influxdata/influxdb/issues/676). Allow storing high precision integer values without losing any information
- [Issue #695](https://github.com/influxdata/influxdb/issues/695). Prevent having duplicate field names in the write payload. (Thanks, @seunglee150)
- [Issue #731](https://github.com/influxdata/influxdb/issues/731). Don't enable the udp plugin if the `enabled` option is set to false
- [Issue #733](https://github.com/influxdata/influxdb/issues/733). Print an `INFO` message when the input plugin is disabled
- [Issue #707](https://github.com/influxdata/influxdb/issues/707). Graphite input plugin should work payload delimited by any whitespace character
- [Issue #734](https://github.com/influxdata/influxdb/issues/734). Don't buffer non replicated writes
- [Issue #465](https://github.com/influxdata/influxdb/issues/465). Recreating a currently deleting db or series doesn't bring back the old data anymore
- [Issue #358](https://github.com/influxdata/influxdb/issues/358). **BREAKING** List series should return as a single series
- [Issue #499](https://github.com/influxdata/influxdb/issues/499). **BREAKING** Querying non-existent database or series will return an error
- [Issue #570](https://github.com/influxdata/influxdb/issues/570). InfluxDB crashes during delete/drop of database
- [Issue #592](https://github.com/influxdata/influxdb/issues/592). Drop series is inefficient

## v0.7.3 [2014-06-13]

### Bugfixes

- [Issue #637](https://github.com/influxdata/influxdb/issues/637). Truncate log files if the last request wasn't written properly
- [Issue #646](https://github.com/influxdata/influxdb/issues/646). CRITICAL: Duplicate shard ids for new shards if old shards are deleted.

## v0.7.2 [2014-05-30]

### Features

- [Issue #521](https://github.com/influxdata/influxdb/issues/521). MODE works on all datatypes (Thanks, @richthegeek)

### Bugfixes

- [Issue #418](https://github.com/influxdata/influxdb/pull/418). Requests or responses larger than MAX_REQUEST_SIZE break things.
- [Issue #606](https://github.com/influxdata/influxdb/issues/606). InfluxDB will fail to start with invalid permission if log.txt didn't exist
- [Issue #602](https://github.com/influxdata/influxdb/issues/602). Merge will fail to work across shards

### Features

## v0.7.1 [2014-05-29]

### Bugfixes

- [Issue #579](https://github.com/influxdata/influxdb/issues/579). Reject writes to nonexistent databases
- [Issue #597](https://github.com/influxdata/influxdb/issues/597). Force compaction after deleting data

### Features

- [Issue #476](https://github.com/influxdata/influxdb/issues/476). Support ARM architecture
- [Issue #578](https://github.com/influxdata/influxdb/issues/578). Support aliasing for expressions in parenthesis
- [Issue #544](https://github.com/influxdata/influxdb/pull/544). Support forcing node removal from a cluster
- [Issue #591](https://github.com/influxdata/influxdb/pull/591). Support multiple udp input plugins (Thanks, @tpitale)
- [Issue #600](https://github.com/influxdata/influxdb/pull/600). Report version, os, arch, and raftName once per day.

## v0.7.0 [2014-05-23]

### Bugfixes

- [Issue #557](https://github.com/influxdata/influxdb/issues/557). Group by time(1y) doesn't work while time(365d) works
- [Issue #547](https://github.com/influxdata/influxdb/issues/547). Add difference function (Thanks, @mboelstra)
- [Issue #550](https://github.com/influxdata/influxdb/issues/550). Fix tests on 32-bit ARM
- [Issue #524](https://github.com/influxdata/influxdb/issues/524). Arithmetic operators and where conditions don't play nice together
- [Issue #561](https://github.com/influxdata/influxdb/issues/561). Fix missing query in parsing errors
- [Issue #563](https://github.com/influxdata/influxdb/issues/563). Add sample config for graphite over udp
- [Issue #537](https://github.com/influxdata/influxdb/issues/537). Incorrect query syntax causes internal error
- [Issue #565](https://github.com/influxdata/influxdb/issues/565). Empty series names shouldn't cause a panic
- [Issue #575](https://github.com/influxdata/influxdb/issues/575). Single point select doesn't interpret timestamps correctly
- [Issue #576](https://github.com/influxdata/influxdb/issues/576). We shouldn't set timestamps and sequence numbers when listing cq
- [Issue #560](https://github.com/influxdata/influxdb/issues/560). Use /dev/urandom instead of /dev/random
- [Issue #502](https://github.com/influxdata/influxdb/issues/502). Fix a
  race condition in assigning id to db+series+field (Thanks @ohurvitz
  for reporting this bug and providing a script to repro)

### Features

- [Issue #567](https://github.com/influxdata/influxdb/issues/567). Allow selecting from multiple series names by separating them with commas (Thanks, @peekeri)

### Deprecated

- [Issue #460](https://github.com/influxdata/influxdb/issues/460). Don't start automatically after installing
- [Issue #529](https://github.com/influxdata/influxdb/issues/529). Don't run influxdb as root
- [Issue #443](https://github.com/influxdata/influxdb/issues/443). Use `name` instead of `username` when returning cluster admins

## v0.6.5 [2014-05-19]

### Features

- [Issue #551](https://github.com/influxdata/influxdb/issues/551). Add TOP and BOTTOM aggregate functions (Thanks, @chobie)

### Bugfixes

- [Issue #555](https://github.com/influxdata/influxdb/issues/555). Fix a regression introduced in the raft snapshot format

## v0.6.4 [2014-05-16]

### Features

- Make the write batch size configurable (also applies to deletes)
- Optimize writing to multiple series
- [Issue #546](https://github.com/influxdata/influxdb/issues/546). Add UDP support for Graphite API (Thanks, @peekeri)

### Bugfixes

- Fix a bug in shard logic that caused short term shards to be clobbered with long term shards
- [Issue #489](https://github.com/influxdata/influxdb/issues/489). Remove replication factor from CreateDatabase command

## v0.6.3 [2014-05-13]

### Features

- [Issue #505](https://github.com/influxdata/influxdb/issues/505). Return a version header with http the response (Thanks, @majst01)
- [Issue #520](https://github.com/influxdata/influxdb/issues/520). Print the version to the log file

### Bugfixes

- [Issue #516](https://github.com/influxdata/influxdb/issues/516). Close WAL log/index files when they aren't being used
- [Issue #532](https://github.com/influxdata/influxdb/issues/532). Don't log graphite connection EOF as an error
- [Issue #535](https://github.com/influxdata/influxdb/issues/535). WAL Replay hangs if response isn't received
- [Issue #538](https://github.com/influxdata/influxdb/issues/538). Don't panic if the same series existed twice in the request with different columns
- [Issue #536](https://github.com/influxdata/influxdb/issues/536). Joining the cluster after shards are creating shouldn't cause new nodes to panic
- [Issue #539](https://github.com/influxdata/influxdb/issues/539). count(distinct()) with fill shouldn't panic on empty groups
- [Issue #534](https://github.com/influxdata/influxdb/issues/534). Create a new series when interpolating

## v0.6.2 [2014-05-09]

### Bugfixes

- [Issue #511](https://github.com/influxdata/influxdb/issues/511). Don't automatically create the database when a db user is created
- [Issue #512](https://github.com/influxdata/influxdb/issues/512). Group by should respect null values
- [Issue #518](https://github.com/influxdata/influxdb/issues/518). Filter Infinities and NaNs from the returned json
- [Issue #522](https://github.com/influxdata/influxdb/issues/522). Committing requests while replaying caused the WAL to skip some log files
- [Issue #369](https://github.com/influxdata/influxdb/issues/369). Fix some edge cases with WAL recovery

## v0.6.1 [2014-05-06]

### Bugfixes

- [Issue #500](https://github.com/influxdata/influxdb/issues/500). Support `y` suffix in time durations
- [Issue #501](https://github.com/influxdata/influxdb/issues/501). Writes with invalid payload should be rejected
- [Issue #507](https://github.com/influxdata/influxdb/issues/507). New cluster admin passwords don't propagate properly to other nodes in a cluster
- [Issue #508](https://github.com/influxdata/influxdb/issues/508). Don't replay WAL entries for servers with no shards
- [Issue #464](https://github.com/influxdata/influxdb/issues/464). Admin UI shouldn't draw graphs for string columns
- [Issue #480](https://github.com/influxdata/influxdb/issues/480). Large values on the y-axis get cut off

## v0.6.0 [2014-05-02]

### Feature

- [Issue #477](https://github.com/influxdata/influxdb/issues/477). Add a udp json interface (Thanks, Julien Ammous)
- [Issue #491](https://github.com/influxdata/influxdb/issues/491). Make initial root password settable through env variable (Thanks, Edward Muller)

### Bugfixes

- [Issue #469](https://github.com/influxdata/influxdb/issues/469). Drop continuous queries when a database is dropped
- [Issue #431](https://github.com/influxdata/influxdb/issues/431). Don't log to standard output if a log file is specified in the config file
- [Issue #483](https://github.com/influxdata/influxdb/issues/483). Return 409 if a database already exist (Thanks, Edward Muller)
- [Issue #486](https://github.com/influxdata/influxdb/issues/486). Columns used in the target of continuous query shouldn't be inserted in the time series
- [Issue #490](https://github.com/influxdata/influxdb/issues/490). Database user password's cannot be changed (Thanks, Edward Muller)
- [Issue #495](https://github.com/influxdata/influxdb/issues/495). Enforce write permissions properly

## v0.5.12 [2014-04-29]

### Bugfixes

- [Issue #419](https://github.com/influxdata/influxdb/issues/419),[Issue #478](https://github.com/influxdata/influxdb/issues/478). Allow hostname, raft and protobuf ports to be changed, without requiring manual intervention from the user

## v0.5.11 [2014-04-25]

### Features

- [Issue #471](https://github.com/influxdata/influxdb/issues/471). Read and write permissions should be settable through the http api

### Bugfixes

- [Issue #323](https://github.com/influxdata/influxdb/issues/323). Continuous queries should guard against data loops
- [Issue #473](https://github.com/influxdata/influxdb/issues/473). Engine memory optimization

## v0.5.10 [2014-04-22]

### Features

- [Issue #463](https://github.com/influxdata/influxdb/issues/463). Allow series names to use any character (escape by wrapping in double quotes)
- [Issue #447](https://github.com/influxdata/influxdb/issues/447). Allow @ in usernames
- [Issue #466](https://github.com/influxdata/influxdb/issues/466). Allow column names to use any character (escape by wrapping in double quotes)

### Bugfixes

- [Issue #458](https://github.com/influxdata/influxdb/issues/458). Continuous queries with group by time() and a column should insert sequence numbers of 1
- [Issue #457](https://github.com/influxdata/influxdb/issues/457). Deleting series that start with capital letters should work

## v0.5.9 [2014-04-18]

### Bugfixes

- [Issue #446](https://github.com/influxdata/influxdb/issues/446). Check for (de)serialization errors
- [Issue #456](https://github.com/influxdata/influxdb/issues/456). Continuous queries failed if one of the group by columns had null value
- [Issue #455](https://github.com/influxdata/influxdb/issues/455). Comparison operators should ignore null values

## v0.5.8 [2014-04-17]

- Renamed config.toml.sample to config.sample.toml

### Bugfixes

- [Issue #244](https://github.com/influxdata/influxdb/issues/244). Reconstruct the query from the ast
- [Issue #449](https://github.com/influxdata/influxdb/issues/449). Heartbeat timeouts can cause reading from connection to lock up
- [Issue #451](https://github.com/influxdata/influxdb/issues/451). Reduce the aggregation state that is kept in memory so that
  aggregation queries over large periods of time don't take insance amount of memory

## v0.5.7 [2014-04-15]

### Features

- Queries are now logged as INFO in the log file before they run

### Bugfixes

- [Issue #328](https://github.com/influxdata/influxdb/issues/328). Join queries with math expressions don't work
- [Issue #440](https://github.com/influxdata/influxdb/issues/440). Heartbeat timeouts in logs
- [Issue #442](https://github.com/influxdata/influxdb/issues/442). shouldQuerySequentially didn't work as expected
  causing count(*) queries on large time series to use
  lots of memory
- [Issue #437](https://github.com/influxdata/influxdb/issues/437). Queries with negative constants don't parse properly
- [Issue #432](https://github.com/influxdata/influxdb/issues/432). Deleted data using a delete query is resurrected after a server restart
- [Issue #439](https://github.com/influxdata/influxdb/issues/439). Report the right location of the error in the query
- Fix some bugs with the WAL recovery on startup

## v0.5.6 [2014-04-08]

### Features

- [Issue #310](https://github.com/influxdata/influxdb/issues/310). Request should support multiple timeseries
- [Issue #416](https://github.com/influxdata/influxdb/issues/416). Improve the time it takes to drop database

### Bugfixes

- [Issue #413](https://github.com/influxdata/influxdb/issues/413). Don't assume that group by interval is greater than a second
- [Issue #415](https://github.com/influxdata/influxdb/issues/415). Include the database when sending an auth error back to the user
- [Issue #421](https://github.com/influxdata/influxdb/issues/421). Make read timeout a config option
- [Issue #392](https://github.com/influxdata/influxdb/issues/392). Different columns in different shards returns invalid results when a query spans those shards

### Bugfixes

## v0.5.5 [2014-04-04]

- Upgrade leveldb 1.10 -> 1.15

  This should be a backward compatible change, but is here for documentation only

### Feature

- Add a command line option to repair corrupted leveldb databases on startup
- [Issue #401](https://github.com/influxdata/influxdb/issues/401). No limit on the number of columns in the group by clause

### Bugfixes

- [Issue #398](https://github.com/influxdata/influxdb/issues/398). Support now() and NOW() in the query lang
- [Issue #403](https://github.com/influxdata/influxdb/issues/403). Filtering should work with join queries
- [Issue #404](https://github.com/influxdata/influxdb/issues/404). Filtering with invalid condition shouldn't crash the server
- [Issue #405](https://github.com/influxdata/influxdb/issues/405). Percentile shouldn't crash for small number of values
- [Issue #408](https://github.com/influxdata/influxdb/issues/408). Make InfluxDB recover from internal bugs and panics
- [Issue #390](https://github.com/influxdata/influxdb/issues/390). Multiple response.WriteHeader when querying as admin
- [Issue #407](https://github.com/influxdata/influxdb/issues/407). Start processing continuous queries only after the WAL is initialized
- Close leveldb databases properly if we couldn't create a new Shard. See leveldb\_shard\_datastore\_test:131

## v0.5.4 [2014-04-02]

### Bugfixes

- [Issue #386](https://github.com/influxdata/influxdb/issues/386). Drop series should work with series containing dots
- [Issue #389](https://github.com/influxdata/influxdb/issues/389). Filtering shouldn't stop prematurely
- [Issue #341](https://github.com/influxdata/influxdb/issues/341). Make the number of shards that are queried in parallel configurable
- [Issue #394](https://github.com/influxdata/influxdb/issues/394). Support count(distinct) and count(DISTINCT)
- [Issue #362](https://github.com/influxdata/influxdb/issues/362). Limit should be enforced after aggregation

## v0.5.3 [2014-03-31]

### Bugfixes

- [Issue #378](https://github.com/influxdata/influxdb/issues/378). Indexing should return if there are no requests added since the last index
- [Issue #370](https://github.com/influxdata/influxdb/issues/370). Filtering and limit should be enforced on the shards
- [Issue #379](https://github.com/influxdata/influxdb/issues/379). Boolean columns should be usable in where clauses
- [Issue #381](https://github.com/influxdata/influxdb/issues/381). Should be able to do deletes as a cluster admin

## v0.5.2 [2014-03-28]

### Bugfixes

- [Issue #342](https://github.com/influxdata/influxdb/issues/342). Data resurrected after a server restart
- [Issue #367](https://github.com/influxdata/influxdb/issues/367). Influxdb won't start if the api port is commented out
- [Issue #355](https://github.com/influxdata/influxdb/issues/355). Return an error on wrong time strings
- [Issue #331](https://github.com/influxdata/influxdb/issues/331). Allow negative time values in the where clause
- [Issue #371](https://github.com/influxdata/influxdb/issues/371). Seris index isn't deleted when the series is dropped
- [Issue #360](https://github.com/influxdata/influxdb/issues/360). Store and recover continuous queries

## v0.5.1 [2014-03-24]

### Bugfixes

- Revert the version of goraft due to a bug found in the latest version

## v0.5.0 [2014-03-24]

### Features

- [Issue #293](https://github.com/influxdata/influxdb/pull/293). Implement a Graphite listener

### Bugfixes

- [Issue #340](https://github.com/influxdata/influxdb/issues/340). Writing many requests while replaying seems to cause commits out of order

## v0.5.0-rc.6 [2014-03-20]

### Bugfixes

- Increase raft election timeout to avoid unecessary relections
- Sort points before writing them to avoid an explosion in the request
  number when the points are written randomly
- [Issue #335](https://github.com/influxdata/influxdb/issues/335). Fixes regexp for interpolating more than one column value in continuous queries
- [Issue #318](https://github.com/influxdata/influxdb/pull/318). Support EXPLAIN queries
- [Issue #333](https://github.com/influxdata/influxdb/pull/333). Fail
  when the password is too short or too long instead of passing it to
  the crypto library

## v0.5.0-rc.5 [2014-03-11]

### Bugfixes

- [Issue #312](https://github.com/influxdata/influxdb/issues/312). WAL should wait for server id to be set before recovering
- [Issue #301](https://github.com/influxdata/influxdb/issues/301). Use ref counting to guard against race conditions in the shard cache
- [Issue #319](https://github.com/influxdata/influxdb/issues/319). Propagate engine creation error correctly to the user
- [Issue #316](https://github.com/influxdata/influxdb/issues/316). Make
  sure we don't starve goroutines if we get an access denied error
  from one of the shards
- [Issue #306](https://github.com/influxdata/influxdb/issues/306). Deleting/Dropping database takes a lot of memory
- [Issue #302](https://github.com/influxdata/influxdb/issues/302). Should be able to set negative timestamps on points
- [Issue #327](https://github.com/influxdata/influxdb/issues/327). Make delete queries not use WAL. This addresses #315, #317 and #314
- [Issue #321](https://github.com/influxdata/influxdb/issues/321). Make sure we split points on shards properly

## v0.5.0-rc.4 [2014-03-07]

### Bugfixes

- [Issue #298](https://github.com/influxdata/influxdb/issues/298). Fix limit when querying multiple shards
- [Issue #305](https://github.com/influxdata/influxdb/issues/305). Shard ids not unique after restart
- [Issue #309](https://github.com/influxdata/influxdb/issues/309). Don't relog the requests on the remote server
- Fix few bugs in the WAL and refactor the way it works (this requires purging the WAL from previous rc)

## v0.5.0-rc.3 [2014-03-03]

### Bugfixes
- [Issue #69](https://github.com/influxdata/influxdb/issues/69). Support column aliases
- [Issue #287](https://github.com/influxdata/influxdb/issues/287). Make the lru cache size configurable
- [Issue #38](https://github.com/influxdata/influxdb/issues/38). Fix a memory leak discussed in this story
- [Issue #286](https://github.com/influxdata/influxdb/issues/286). Make the number of open shards configurable
- Make LevelDB use the max open files configuration option.

## v0.5.0-rc.2 [2014-02-27]

### Bugfixes

- [Issue #274](https://github.com/influxdata/influxdb/issues/274). Crash after restart
- [Issue #277](https://github.com/influxdata/influxdb/issues/277). Ensure duplicate shards won't be created
- [Issue #279](https://github.com/influxdata/influxdb/issues/279). Limits not working on regex queries
- [Issue #281](https://github.com/influxdata/influxdb/issues/281). `./influxdb -v` should print the sha when building from source
- [Issue #283](https://github.com/influxdata/influxdb/issues/283). Dropping shard and restart in cluster causes panic.
- [Issue #288](https://github.com/influxdata/influxdb/issues/288). Sequence numbers should be unique per server id

## v0.5.0-rc.1 [2014-02-25]

### Bugfixes

- Ensure large deletes don't take too much memory
- [Issue #240](https://github.com/influxdata/influxdb/pull/240). Unable to query against columns with `.` in the name.
- [Issue #250](https://github.com/influxdata/influxdb/pull/250). different result between normal and continuous query with "group by" clause
- [Issue #216](https://github.com/influxdata/influxdb/pull/216). Results with no points should exclude columns and points

### Features

- [Issue #243](https://github.com/influxdata/influxdb/issues/243). Should have endpoint to GET a user's attributes.
- [Issue #269](https://github.com/influxdata/influxdb/pull/269), [Issue #65](https://github.com/influxdata/influxdb/issues/65) New clustering architecture (see docs), with the side effect that queries can be distributed between multiple shards
- [Issue #164](https://github.com/influxdata/influxdb/pull/269),[Issue #103](https://github.com/influxdata/influxdb/pull/269),[Issue #166](https://github.com/influxdata/influxdb/pull/269),[Issue #165](https://github.com/influxdata/influxdb/pull/269),[Issue #132](https://github.com/influxdata/influxdb/pull/269) Make request log a log file instead of leveldb with recovery on startup

### Deprecated

- [Issue #189](https://github.com/influxdata/influxdb/issues/189). `/cluster_admins` and `/db/:db/users` return usernames in a `name` key instead of `username` key.
- [Issue #216](https://github.com/influxdata/influxdb/pull/216). Results with no points should exclude columns and points

## v0.4.4 [2014-02-05]

### Features

- Make the leveldb max open files configurable in the toml file

## v0.4.3 [2014-01-31]

### Bugfixes

- [Issue #225](https://github.com/influxdata/influxdb/issues/225). Remove a hard limit on the points returned by the datastore
- [Issue #223](https://github.com/influxdata/influxdb/issues/223). Null values caused count(distinct()) to panic
- [Issue #224](https://github.com/influxdata/influxdb/issues/224). Null values broke replication due to protobuf limitation

## v0.4.1 [2014-01-30]

### Features

- [Issue #193](https://github.com/influxdata/influxdb/issues/193). Allow logging to stdout. Thanks @schmurfy
- [Issue #190](https://github.com/influxdata/influxdb/pull/190). Add support for SSL.
- [Issue #194](https://github.com/influxdata/influxdb/pull/194). Should be able to disable Admin interface.

### Bugfixes

- [Issue #33](https://github.com/influxdata/influxdb/issues/33). Don't call WriteHeader more than once per request
- [Issue #195](https://github.com/influxdata/influxdb/issues/195). Allow the bind address to be configurable, Thanks @schmurfy.
- [Issue #199](https://github.com/influxdata/influxdb/issues/199). Make the test timeout configurable
- [Issue #200](https://github.com/influxdata/influxdb/issues/200). Selecting `time` or `sequence_number` silently fail
- [Issue #215](https://github.com/influxdata/influxdb/pull/215). Server fails to start up after Raft log compaction and restart.

## v0.4.0 [2014-01-17]

## Features

- [Issue #86](https://github.com/influxdata/influxdb/issues/86). Support arithmetic expressions in select clause
- [Issue #92](https://github.com/influxdata/influxdb/issues/92). Change '==' to '=' and '!=' to '<>'
- [Issue #88](https://github.com/influxdata/influxdb/issues/88). Support datetime strings
- [Issue #64](https://github.com/influxdata/influxdb/issues/64). Shard writes and queries across cluster with replay for briefly downed nodes (< 24 hrs)
- [Issue #78](https://github.com/influxdata/influxdb/issues/78). Sequence numbers persist across restarts so they're not reused
- [Issue #102](https://github.com/influxdata/influxdb/issues/102). Support expressions in where condition
- [Issue #101](https://github.com/influxdata/influxdb/issues/101). Support expressions in aggregates
- [Issue #62](https://github.com/influxdata/influxdb/issues/62). Support updating and deleting column values
- [Issue #96](https://github.com/influxdata/influxdb/issues/96). Replicate deletes in a cluster
- [Issue #94](https://github.com/influxdata/influxdb/issues/94). delete queries
- [Issue #116](https://github.com/influxdata/influxdb/issues/116). Use proper logging
- [Issue #40](https://github.com/influxdata/influxdb/issues/40). Use TOML instead of JSON in the config file
- [Issue #99](https://github.com/influxdata/influxdb/issues/99). Support list series in the query language
- [Issue #149](https://github.com/influxdata/influxdb/issues/149). Cluster admins should be able to perform reads and writes.
- [Issue #108](https://github.com/influxdata/influxdb/issues/108). Querying one point using `time =`
- [Issue #114](https://github.com/influxdata/influxdb/issues/114). Servers should periodically check that they're consistent.
- [Issue #93](https://github.com/influxdata/influxdb/issues/93). Should be able to drop a time series
- [Issue #177](https://github.com/influxdata/influxdb/issues/177). Support drop series in the query language.
- [Issue #184](https://github.com/influxdata/influxdb/issues/184). Implement Raft log compaction.
- [Issue #153](https://github.com/influxdata/influxdb/issues/153). Implement continuous queries

### Bugfixes

- [Issue #90](https://github.com/influxdata/influxdb/issues/90). Group by multiple columns panic
- [Issue #89](https://github.com/influxdata/influxdb/issues/89). 'Group by' combined with 'where' not working
- [Issue #106](https://github.com/influxdata/influxdb/issues/106). Don't panic if we only see one point and can't calculate derivative
- [Issue #105](https://github.com/influxdata/influxdb/issues/105). Panic when using a where clause that reference columns with null values
- [Issue #61](https://github.com/influxdata/influxdb/issues/61). Remove default limits from queries
- [Issue #118](https://github.com/influxdata/influxdb/issues/118). Make column names starting with '_' legal
- [Issue #121](https://github.com/influxdata/influxdb/issues/121). Don't fall back to the cluster admin auth if the db user auth fails
- [Issue #127](https://github.com/influxdata/influxdb/issues/127). Return error on delete queries with where condition that don't have time
- [Issue #117](https://github.com/influxdata/influxdb/issues/117). Fill empty groups with default values
- [Issue #150](https://github.com/influxdata/influxdb/pull/150). Fix parser for when multiple divisions look like a regex.
- [Issue #158](https://github.com/influxdata/influxdb/issues/158). Logged deletes should be stored with the time range if missing.
- [Issue #136](https://github.com/influxdata/influxdb/issues/136). Make sure writes are replicated in order to avoid triggering replays
- [Issue #145](https://github.com/influxdata/influxdb/issues/145). Server fails to join cluster if all starting at same time.
- [Issue #176](https://github.com/influxdata/influxdb/issues/176). Drop database should take effect on all nodes
- [Issue #180](https://github.com/influxdata/influxdb/issues/180). Column names not returned when running multi-node cluster and writing more than one point.
- [Issue #182](https://github.com/influxdata/influxdb/issues/182). Queries with invalid limit clause crash the server

### Deprecated

- deprecate '==' and '!=' in favor of '=' and '<>', respectively
- deprecate `/dbs` (for listing databases) in favor of a more consistent `/db` endpoint
- deprecate `username` field for a more consistent `name` field in `/db/:db/users` and `/cluster_admins`
- deprecate endpoints `/db/:db/admins/:user` in favor of using `/db/:db/users/:user` which should
  be used to update user flags, password, etc.
- Querying for column names that don't exist no longer throws an error.

## v0.3.2

## Features

- [Issue #82](https://github.com/influxdata/influxdb/issues/82). Add endpoint for listing available admin interfaces.
- [Issue #80](https://github.com/influxdata/influxdb/issues/80). Support durations when specifying start and end time
- [Issue #81](https://github.com/influxdata/influxdb/issues/81). Add support for IN

## Bugfixes

- [Issue #75](https://github.com/influxdata/influxdb/issues/75). Don't allow time series names that start with underscore
- [Issue #85](https://github.com/influxdata/influxdb/issues/85). Non-existing columns exist after they have been queried before

## v0.3.0

## Features

- [Issue #51](https://github.com/influxdata/influxdb/issues/51). Implement first and last aggregates
- [Issue #35](https://github.com/influxdata/influxdb/issues/35). Support table aliases in Join Queries
- [Issue #71](https://github.com/influxdata/influxdb/issues/71). Add WillReturnSingleSeries to the Query
- [Issue #61](https://github.com/influxdata/influxdb/issues/61). Limit should default to 10k
- [Issue #59](https://github.com/influxdata/influxdb/issues/59). Add histogram aggregate function

## Bugfixes

- Fix join and merges when the query is a descending order query
- [Issue #57](https://github.com/influxdata/influxdb/issues/57). Don't panic when type of time != float
- [Issue #63](https://github.com/influxdata/influxdb/issues/63). Aggregate queries should not have a sequence_number column

## v0.2.0

### Features

- [Issue #37](https://github.com/influxdata/influxdb/issues/37). Support the negation of the regex matcher !~
- [Issue #47](https://github.com/influxdata/influxdb/issues/47). Spill out query and database detail at the time of bug report

### Bugfixes

- [Issue #36](https://github.com/influxdata/influxdb/issues/36). The regex operator should be =~ not ~=
- [Issue #39](https://github.com/influxdata/influxdb/issues/39). Return proper content types from the http api
- [Issue #42](https://github.com/influxdata/influxdb/issues/42). Make the api consistent with the docs
- [Issue #41](https://github.com/influxdata/influxdb/issues/41). Table/Points not deleted when database is dropped
- [Issue #45](https://github.com/influxdata/influxdb/issues/45). Aggregation shouldn't mess up the order of the points
- [Issue #44](https://github.com/influxdata/influxdb/issues/44). Fix crashes on RHEL 5.9
- [Issue #34](https://github.com/influxdata/influxdb/issues/34). Ascending order always return null for columns that have a null value
- [Issue #55](https://github.com/influxdata/influxdb/issues/55). Limit should limit the points that match the Where clause
- [Issue #53](https://github.com/influxdata/influxdb/issues/53). Writing null values via HTTP API fails

### Deprecated

- Preparing to deprecate `/dbs` (for listing databases) in favor of a more consistent `/db` endpoint
- Preparing to deprecate `username` field for a more consistent `name` field in the `/db/:db/users`
- Preparing to deprecate endpoints `/db/:db/admins/:user` in favor of using `/db/:db/users/:user` which should
  be used to update user flags, password, etc.

## v0.1.0

### Features

- [Issue #29](https://github.com/influxdata/influxdb/issues/29). Semicolon is now optional in queries
- [Issue #31](https://github.com/influxdata/influxdb/issues/31). Support Basic Auth as well as query params for authentication.

### Bugfixes

- Don't allow creating users with empty username
- [Issue #22](https://github.com/influxdata/influxdb/issues/22). Don't set goroot if it was set
- [Issue #25](https://github.com/influxdata/influxdb/issues/25). Fix queries that use the median aggregator
- [Issue #26](https://github.com/influxdata/influxdb/issues/26). Default log and db directories should be in /opt/influxdb/shared/data
- [Issue #27](https://github.com/influxdata/influxdb/issues/27). Group by should not blow up if the one of the columns in group by has null values
- [Issue #30](https://github.com/influxdata/influxdb/issues/30). Column indexes/names getting off somehow
- [Issue #32](https://github.com/influxdata/influxdb/issues/32). Fix many typos in the codebase. Thanks @pborreli

## v0.0.9

#### Features

- Add stddev(...) support
- Better docs, thanks @auxesis and @d-snp.

#### Bugfixes

- Set PYTHONPATH and CC appropriately on mac os x.
- [Issue #18](https://github.com/influxdata/influxdb/issues/18). Fix 386 debian and redhat packages
- [Issue #23](https://github.com/influxdata/influxdb/issues/23). Fix the init scripts on redhat

## v0.0.8

#### Features

- Add a way to reset the root password from the command line.
- Add distinct(..) and derivative(...) support
- Print test coverage if running go1.2

#### Bugfixes

- Fix the default admin site path in the .deb and .rpm packages.
- Fix the configuration filename in the .tar.gz package.

## v0.0.7

#### Features

- include the admin site in the repo to make it easier for newcomers.

## v0.0.6

#### Features

- Add count(distinct(..)) support

#### Bugfixes

- Reuse levigo read/write options.

## v0.0.5

#### Features

- Cache passwords in memory to speed up password verification
- Add MERGE and INNER JOIN support

#### Bugfixes

- All columns should be returned if `select *` was used
- Read/Write benchmarks

## v0.0.2

#### Features

- Add an admin UI
- Deb and RPM packages

#### Bugfixes

- Fix some nil pointer dereferences
- Cleanup the aggregators implementation

## v0.0.1 [2013-10-22]

  * Initial Release
