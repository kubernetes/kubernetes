# Exporting cAdvisor Stats to Kafka

cAdvisor supports exporting stats to [Kafka](http://kafka.apache.org/). To use Kafka, you need to provide the additional flags to cAdvisor:

Set the storage driver as Kafka:

```
 -storage_driver=kafka
```

If no broker are provided it will default to a broker listening at localhost:9092, with 'stats' as the default topic.


Specify a Kafka broker address:

```
-storage_driver_kafka_broker_list=localhost:9092

```

Specify a Kafka topic:

```
-storage_driver_kafka_topic=myTopic
```

As of version 9.0. Kafka supports TLS client auth:

```
 # To enable TLS client auth support you need to provide the following:

 # Location to Certificate Authority certificate
  -storage_driver_kafka_ssl_ca=/path/to/ca.pem

 # Location to client certificate certificate
  -storage_driver_kafka_ssl_cert=/path/to/client_cert.pem

 # Location to client certificate key
  -storage_driver_kafka_ssl_key=/path/to/client_key.pem

 # Verify SSL certificate chain (default: true)
  -storage_driver_kafka_ssl_verify=false
```
