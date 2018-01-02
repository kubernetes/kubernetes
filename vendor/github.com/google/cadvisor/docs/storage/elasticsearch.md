# Exporting cAdvisor Stats to ElasticSearch

cAdvisor supports exporting stats to [ElasticSearch](https://www.elastic.co/). To use ES, you need to provide the additional flags to cAdvisor:

Set the storage driver as ES:

```
 -storage_driver=elasticsearch
```

Specify ES host address:

```
 -storage_driver_es_host="http://elasticsearch:9200"
```

There are also optional flags:

```
 # ElasticSearch type name. By default it's "stats".
 -storage_driver_es_type="stats"
 # ElasticSearch can use a sniffing process to find all nodes of your cluster automatically. False by default.
 -storage_driver_es_enable_sniffer=false
```

# Examples

For a detailed tutorial, see [docker-elk-cadvisor-dashboards](https://github.com/gregbkr/docker-elk-cadvisor-dashboards)
