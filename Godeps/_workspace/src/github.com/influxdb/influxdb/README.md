# InfluxDB [![Circle CI](https://circleci.com/gh/influxdb/influxdb/tree/master.svg?style=svg)](https://circleci.com/gh/influxdb/influxdb/tree/master)

## An Open-Source, Distributed, Time Series Database

> InfluxDB v0.9.0 is now out. Going forward, the 0.9.x series of releases will not make breaking API changes or breaking changes to the underlying data storage. However, 0.9.x clustering should be considered an alpha release.

InfluxDB is an open source **distributed time series database** with
**no external dependencies**. It's useful for recording metrics,
events, and performing analytics.

## Features

* Built-in [HTTP API](https://docs.influxdata.com/influxdb/v0.9/guides/writing_data/) so you don't have to write any server side code to get up and running.
* Data can be tagged, allowing very flexible querying.
* SQL-like query language.
* Clustering is supported out of the box, so that you can scale horizontally to handle your data.
* Simple to install and manage, and fast to get data in and out.
* It aims to answer queries in real-time. That means every data point is
  indexed as it comes in and is immediately available in queries that
  should return in < 100ms.

## Getting Started
*The following directions apply only to the 0.9.x series or building from the source on master.*

### Building

You don't need to build the project to use it - you can use any of our
[pre-built packages](https://influxdata.com/downloads/) to install InfluxDB. That's
the recommended way to get it running. However, if you want to contribute to the core of InfluxDB, you'll need to build.
For those adventurous enough, you can
[follow along on our docs](http://github.com/influxdb/influxdb/blob/master/CONTRIBUTING.md).

### Starting InfluxDB
* `service influxdb start` if you have installed InfluxDB using an official Debian or RPM package.
* `systemctl start influxdb` if you have installed InfluxDB using an official Debian or RPM package, and are running a distro with `systemd`. For example, Ubuntu 15 or later.
* `$GOPATH/bin/influxd` if you have built InfluxDB from source.

### Creating your first database

```
curl -G 'http://localhost:8086/query' --data-urlencode "q=CREATE DATABASE mydb"
```

### Insert some data
```
curl -XPOST 'http://localhost:8086/write?db=mydb' \
-d 'cpu,host=server01,region=uswest load=42 1434055562000000000'

curl -XPOST 'http://localhost:8086/write?db=mydb' \
-d 'cpu,host=server02,region=uswest load=78 1434055562000000000'

curl -XPOST 'http://localhost:8086/write?db=mydb' \
-d 'cpu,host=server03,region=useast load=15.4 1434055562000000000'
```

### Query for the data
```JSON
curl -G http://localhost:8086/query?pretty=true --data-urlencode "db=mydb" \
--data-urlencode "q=SELECT * FROM cpu WHERE host='server01' AND time < now() - 1d"
```

### Analyze the data
```JSON
curl -G http://localhost:8086/query?pretty=true --data-urlencode "db=mydb" \
--data-urlencode "q=SELECT mean(load) FROM cpu WHERE region='uswest'"
```

## Helpful Links

* Understand the [design goals and motivations of the project](https://docs.influxdata.com/influxdb/v0.9/introduction/overview/).
* Follow the [getting started guide](https://docs.influxdata.com/influxdb/v0.9/introduction/getting_started/) to find out how to install InfluxDB, start writing more data, and issue more queries - in just a few minutes.
* See the  [HTTP API documentation to start writing a library for your favorite language](https://docs.influxdata.com/influxdb/v0.9/guides/writing_data/).

## Looking for Support?

InfluxDB has technical support subscriptions to help your project succeed. We offer Developer Support for organizations in active development and Production Support for companies requiring the best response times and SLAs on technical fixes. Visit our [support page](https://influxdata.com/services/) to learn which subscription is right for you, or contact sales@influxdb.com for a quote.
