# `influx_stress` usage and configuration

The binary for `influx_stress` comes bundled with all influx installations. 
To run it against an `influxd` instance located at  `localhost:8086` with the default configuration options:

See more about the [default configuration options](https://github.com/influxdata/influxdb/blob/master/stress/stress.toml)

```bash
$ influx_stress
```

To run `influx_stress` with a configuration file:
```bash
$ influx_stress -config my_awesome_test.toml
```

To daemonize `influx_stress` and save the output to a results file:
```bash
$ influx_stress -config my_awesome_test.toml > my_awesome_test_out.txt 2>&1 &
```

To run multiple instances of `influx_stress` just change the `measurement` each test writes to, details below
```bash
$ influx_stress -config my_awesome_test1.toml > my_awesome_test_out1.txt 2>&1 &
$ influx_stress -config my_awesome_test2.toml > my_awesome_test_out2.txt 2>&1 &
```

Below is a sample configuration file with comments explaining the different options
```toml
# The [provision] section creates a new database on the target instance for the stress test to write points to and perform queries against
# This section can be deleted if the instance is manually configured. In that case make sure that the database referenced in [write] exists
# The provisioner will try to delete the database before trying to recreate it.

[provision]
  [provision.basic]
    # If set to false you can delete this section from the config
    enabled = true
    # address of the node to be provisioned
    address = "<node1_ip>:8086"
    # name of the database to create
    database = "stress"
    # This must be set to true
    reset_database = true

# The [write] section defines the shape of the generated data and configures the InfluxDB client
[write]
  # The [write.point_generator] defines the shape of the generated data
  [write.point_generator]
    [write.point_generator.basic]
      # This needs to be set to true
      enabled = true
      # The total number of points a stress_test will write is determined by multiplying the following two numbers:
      # point_count * series_count = total_points
      # Number of points to write to the database for each series
      point_count = 100
      # Number of series to write to the database?
      series_count = 100000
      # This simulates collection interval in the timestamps of generated points
      tick = "10s"
      # This must be set to true
      jitter = true
      # The measurement name for the generated points
      measurement = "cpu"
      # The generated timestamps follow the pattern of { start_date + (n * tick) }
      # This sequence is preserved for each series and is always increasing
      start_date = "2006-Jan-02"
      # Precision for generated points
      # This setting MUST be the same as [write.influx_client.basic]precision
      precision = "s"
      # The '[[]]' in toml format indicates that the element is an array of items. 
      # [[write.point_generator.basic.tag]] defines a tag on the generated points
      # key is the tag key
      # value is the tag value
      # The first tag defined will have '-0' through '-{series_count}' added to the end of the string
      [[write.point_generator.basic.tag]]
        key = "host"
        value = "server"
      [[write.point_generator.basic.tag]]
        key = "location"
        value = "us-west"
      # [[write.point_generator.basic.field]] defines a field on the generated points
      # key is the field key
      # value is the type of the field
      [[write.point_generator.basic.field]]
        key = "value"
        # Can be either "float64", "int", "bool"
        value = "float64"

  # The [write.influx_client] defines what influx instances the stress_test targets
  [write.influx_client]
    [write.influx_client.basic]
      # This must be set to true
      enabled = true
      # This is an array of addresses
      # addresses = ["<node1_ip>:8086","<node2_ip>:8086","<node3_ip>:8086"] to target a cluster
      addresses = ["<node1_ip>:8086"] # to target an individual node 
      # This database in the in the target influx instance to write to
      # This database MUST be created in the target instance or the test will fail
      database = "stress"
      # Write precision for points
      # This setting MUST be the same as [write.point_generator.basic]precision
      precision = "s"
      # The number of point to write to the database with each POST /write sent
      batch_size = 5000
      # An optional amount of time for a worker to wait between POST requests
      batch_interval = "0s"
      # The number of workers to use to write to the database
      # More workers == more load with diminishing returns starting at ~5 workers
      # 10 workers provides a medium-high level of load to the database
      concurrency = 10
      # This must be set to false
      ssl = false
      # This must be set to "line_http"
      format = "line_http"
```