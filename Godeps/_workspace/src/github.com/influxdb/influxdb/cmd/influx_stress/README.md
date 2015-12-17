# `influx_stress`

## Ways to run

### `influx_stress`
This runs a basic stress test with the [default config](https://github.com/influxdb/influxdb/blob/master/stress/stress.toml) For more information on the configuration file please see the default. For additional questions please contact @mjdesa

### `influx_stress -config someConfig.toml`
This runs the stress test with a valid configuration file located at `someConfig.tom`

## Flags

If flags are defined they overwrite the config from any file passed in.

### `-addr` string
IP address and port of database where response times will persist (e.g., localhost:8086)

`default` = "http://localhost:8086"

### `-config` string
The relative path to the stress test configuration file.

`default` = [config](https://github.com/influxdb/influxdb/blob/master/stress/stress.toml)

### `-cpuprofile` filename
Writes the result of Go's cpu profile to filename

`default` = no profiling

### `-database` string
Name of database on `-addr` that `influx_stress` will persist write and query response times

`default` = "stress"

### `-tags` value
A comma separated list of tags to add to write and query response times.

`default` = ""