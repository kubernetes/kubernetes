# Import/Export

## Exporting from 0.8.9

Version `0.8.9` of InfluxDB adds support to export your data to a format that can be imported into `0.9.3` and later.

Note that `0.8.9` can be found here:

```
http://get.influxdb.org.s3.amazonaws.com/influxdb_0.8.9_amd64.deb
http://get.influxdb.org.s3.amazonaws.com/influxdb-0.8.9-1.x86_64.rpm
```

### Design

`0.8.9` exports raw data to a flat file that includes two sections, `DDL` and `DML`.  You can choose to export them independently (see below).

The `DDL` section contains the sql commands to create databases and retention policies.  the `DML` section is [line protocol](https://github.com/influxdb/influxdb/blob/master/tsdb/README.md) and can be directly posted to the [http endpoint](https://influxdb.com/docs/v0.9/guides/writing_data.html) in `0.9`.  Remember that batching is important and we don't recommend batch sizes over 5k.

You need to specify a database and shard group when you export.

To list out your shards, use the following http endpoint:

`/cluster/shard_spaces`

example:
```sh
http://username:password@localhost:8086/cluster/shard_spaces
```

Then, to export a database with then name "metrics" and a shard space with the name "default", issue the following curl command:

```sh
curl -o export http://username:password@http://localhost:8086/export/metrics/default
```

Compression is supported, and will result in a significantly smaller file size.

Use the following command for compression:
```sh
curl -o export.gz --compressed http://username:password@http://localhost:8086/export/metrics/default
```

You can also export just the `DDL` with this option:

```sh
curl -o export.ddl http://username:password@http://localhost:8086/export/metrics/default?l=ddl
```

Or just the `DML` with this option:

```sh
curl -o export.dml.gz --compressed http://username:password@http://localhost:8086/export/metrics/default?l=dml
```

### Assumptions

- Series name mapping follows these [guidelines](https://influxdb.com/docs/v0.8/advanced_topics/schema_design.html)
- Database name will map directly from `0.8` to `0.9`
- Shard Spaces map to Retention Policies
- Shard Space Duration is ignored, as in `0.9` we determine shard size automatically
- Regex is used to match the correct series names and only exports that data for the database
- Duration becomes the new Retention Policy duration

- Users are not migrated due to inability to get passwords.  Anyone using users will need to manually set these back up in `0.9`

### Upgrade Recommendations

It's recommended that you upgrade to `0.9.3` first and have all your writes going there.  Then, on the `0.8.X` instances, upgrade to `0.8.9`.

It is important that when exporting you change your config to allow for the http endpoints not timing out.  To do so, make this change in your config:

```toml
# Configure the http api
[api]
read-timeout = "0s"
```

### Exceptions

If a series can't be exported to tags based on the guidelines mentioned above,
we will insert the entire series name as the measurement name.  You can either 
allow that to import into the new InfluxDB instance, or you can do your own 
data massage on it prior to importing it.

For example, if you have the following series name:

```
metric.disk.c.host.server01.single
```

It will export as exactly thta as the measurement name and no tags:

```
metric.disk.c.host.server01.single
```

### Export Metrics

When you export, you will now get comments inline in the `DML`:

`# Found 999 Series for export`

As well as count totals for each series exported:

`# Series FOO - Points Exported: 999`

With a total at the bottom:

`# Points Exported: 999`

You can grep the file that was exported at the end to get all the export metrics:

`cat myexport | grep Exported`

## Importing

Version `0.9.3` of InfluxDB adds support to import your data from version `0.8.9`.

## Caveats

For the export/import to work, all requisites have to be met.  For export, all series names in `0.8` should be in the following format:

```
<tagName>.<tagValue>.<tagName>.<tagValue>.<measurement>
```
for example:
```
az.us-west-1.host.serverA.cpu
```
or any number of tags 
```
building.2.temperature
```

Additionally, the fields need to have a consistent type (all float64, int64, etc) for every write in `0.8`.  Otherwise they have the potential to fail writes in the import.
See below for more information.

## Running the import command
 
 To import via the cli, you can specify the following command:

 ```sh
 influx -import -path=metrics-default.gz -compressed
 ```

 If the file is not compressed you can issue it without the `-compressed` flag:

 ```sh
 influx -import -path=metrics-default
 ```

 To redirect failed import lines to another file, run this command:

 ```sh
 influx -import -path=metrics-default.gz -compressed > failures
 ```

 The import will use the line protocol in batches of 5,000 lines per batch when sending data to the server.
 
### Throttiling the import
 
 If you need to throttle the import so the database has time to ingest, you can use the `-pps` flag.  This will limit the points per second that will be sent to the server.
 
  ```sh
 influx -import -path=metrics-default.gz -compressed -pps 50000 > failures
 ```
 
 Which is stating that you don't want MORE than 50,000 points per second to write to the database. Due to the processing that is taking place however, you will likely never get exactly 50,000 pps, more like 35,000 pps, etc. 

## Understanding the results of the import

During the import, a status message will write out for every 100,000 points imported and report stats on the progress of the import:

```
2015/08/21 14:48:01 Processed 3100000 lines.  Time elapsed: 56.740578415s.  Points per second (PPS): 54634
```

 The batch will give some basic stats when finished:

 ```sh
 2015/07/29 23:15:20 Processed 2 commands
 2015/07/29 23:15:20 Processed 70207923 inserts
 2015/07/29 23:15:20 Failed 29785000 inserts
 ```

 Most inserts fail due to the following types of error:

 ```sh
 2015/07/29 22:18:28 error writing batch:  write failed: field type conflict: input field "value" on measurement "metric" is type float64, already exists as type integer
 ```

 This is due to the fact that in `0.8` a field could get created and saved as int or float types for independent writes.  In `0.9` the field has to have a consistent type.
