package stress

var s = `
[provision]
  [provision.basic]
    enabled = true
    address = "localhost:8086"
    database = "stress"
    reset_database = true

[write]
  [write.point_generator]
    [write.point_generator.basic]
      enabled = true
      point_count = 100
      series_count = 100000
      tick = "10s"
      jitter = true
      measurement = "cpu"
      start_date = "2006-Jan-02"
      precision = "n"
      [[write.point_generator.basic.tag]]
        key = "host"
        value = "server"
      [[write.point_generator.basic.tag]]
        key = "location"
        value = "us-west"
      [[write.point_generator.basic.field]]
        key = "value"
        value = "float64"


  [write.influx_client]
    [write.influx_client.basic]
      enabled = true
      addresses = ["localhost:8086"]
      database = "stress"
      precision = "n"
      batch_size = 5000
      batch_interval = "0s"
      concurrency = 10
      ssl = false
      format = "line_http" # line_udp, graphite_tcp, graphite_udp

[read]
  [read.query_generator]
    [read.query_generator.basic]
      template = "SELECT count(value) FROM cpu where host='server-%v'"
      query_count = 250

  [read.query_client]
    [read.query_client.basic]
      enabled = true
      addresses = ["localhost:8086"]
      database = "stress"
      query_interval = "100ms"
      concurrency = 1
`

// BasicStress returns a config for a basic
// stress test.
func BasicStress() (*Config, error) {
	return DecodeConfig(s)
}
