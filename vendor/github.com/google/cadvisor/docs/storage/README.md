# cAdvisor Storage Plugins

cAdvisor supports exporting stats to various storage driver plugins. To enable a storage driver, set the `-storage_driver` flag.

## Storage drivers

- [BigQuery](https://cloud.google.com/bigquery/). See the [documentation](../../storage/bigquery/README.md) for usage.
- [ElasticSearch](https://www.elastic.co/). See the [documentation](elasticsearch.md) for usage and examples.
- [InfluxDB](https://influxdb.com/). See the [documentation](influxdb.md) for usage and examples.
- [Kafka](http://kafka.apache.org/). See the [documentation](kafka.md) for usage.
- [Prometheus](https://prometheus.io). See the [documentation](prometheus.md) for usage and examples.
- [Redis](http://redis.io/)
- [StatsD](https://github.com/etsy/statsd)
- `stdout` - write stats to standard output.
