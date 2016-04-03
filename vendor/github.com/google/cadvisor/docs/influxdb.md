# Exporting cAdvisor Stats to InfluxDB

cAdvisor supports exporting stats to [InfluxDB](http://influxdb.com). To use InfluxDB, you need to pass some additional flags to cAdvisor telling it where the InfluxDB instance is located:

Set the storage driver as InfluxDB.

```
 -storage_driver=influxdb
```

Specify what InfluxDB instance to push data to:

```
 # The *ip:port* of the database. Default is 'localhost:8086'
 -storage_driver_host=ip:port
 # database name. Uses db 'cadvisor' by default
 -storage_driver_db
 # database username. Default is 'root'
 -storage_driver_user
 # database password. Default is 'root'
 -storage_driver_password
 # Use secure connection with database. False by default
 -storage_driver_secure
```

# Examples

[Brian Christner](https://www.brianchristner.io) wrote a detailed post on [setting up Docker monitoring](https://www.brianchristner.io/how-to-setup-docker-monitoring) with cAdvisor and Influxdb.  A docker compose configuration for setting up cadvisor-influxdb-grafana can be found [here](https://github.com/dalekurt/docker-monitoring/blob/master/docker-compose.yml).
