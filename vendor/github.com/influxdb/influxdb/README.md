# InfluxDB [![Circle CI](https://circleci.com/gh/influxdb/influxdb/tree/master.svg?style=svg)](https://circleci.com/gh/influxdb/influxdb/tree/master)

## An Open-Source, Distributed, Time Series Database

> InfluxDB v0.9.0 is now out. Going forward, the 0.9.x series of releases will not make breaking API changes or breaking changes to the underlying data storage. However, 0.9.0 clustering should be considered an alpha release.

InfluxDB is an open source **distributed time series database** with
**no external dependencies**. It's useful for recording metrics,
events, and performing analytics.

## Features

* Built-in [HTTP API](http://influxdb.com/docs/v0.9/concepts/reading_and_writing_data.html) so you don't have to write any server side code to get up and running.
* Clustering is supported out of the box, so that you can scale horizontally to handle your data.
* Simple to install and manage, and fast to get data in and out.
* It aims to answer queries in real-time. That means every data point is
  indexed as it comes in and is immediately available in queries that
  should return in < 100ms.

## Getting Started
*The following directions apply only to the 0.9.0 release or building from the source on master.*

### Building

You don't need to build the project to use it - you can use any of our
[pre-built packages](http://influxdb.com/download/index.html) to install InfluxDB. That's
the recommended way to get it running. However, if you want to contribute to the core of InfluxDB, you'll need to build.
For those adventurous enough, you can
[follow along on our docs](http://github.com/influxdb/influxdb/blob/master/CONTRIBUTING.md).

### Starting InfluxDB
* `service influxdb start` if you have installed InfluxDB using an official Debian or RPM package.
* `$GOPATH/bin/influxd` if you have built InfluxDB from source.

### Creating your first database

```
curl -G 'http://localhost:8086/query' --data-urlencode "q=CREATE DATABASE mydb"
```

### Insert some data
```
curl -XPOST 'http://localhost:8086/write?db=mydb' \
-d 'cpu,host=server01,region=uswest value=1.0 1434055562000000000'
```

### Query for the data
```JSON
curl -G http://localhost:8086/query?pretty=true --data-urlencode "db=mydb" \
--data-urlencode "q=SELECT * FROM cpu WHERE host='server01'"
```
## Helpful Links

* Understand the [design goals and motivations of the project](http://influxdb.com/docs/v0.9/introduction/overview.html).
* Follow the [getting started guide](http://influxdb.com/docs/v0.9/introduction/getting_started.html) to find out how to install InfluxDB, start writing more data, and issue more queries - in just a few minutes.
* See the  [HTTP API documentation to start writing a library for your favorite language](http://influxdb.com/docs/v0.9/concepts/reading_and_writing_data.html).
