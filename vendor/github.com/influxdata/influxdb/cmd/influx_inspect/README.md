# `influx_inspect`

## Ways to run

### `influx_inspect`
Will print usage for the tool.

### `influx_inspect report`
Displays series meta-data for all shards.  Default location [$HOME/.influxdb]

### `influx_inspect dumptsm`
Dumps low-level details about tsm1 files

#### Flags

##### `-index` bool
Dump raw index data.

`default` = false

#### `-blocks` bool
Dump raw block data.

`default` = false

#### `-all`
Dump all data. Caution: This may print a lot of information.

`default` = false

#### `-filter-key`
Only display index and block data match this key substring.

`default` = ""


### `influx_inspect export`
Exports all tsm files to line protocol.  This output file can be imported via the [influx](https://github.com/influxdata/influxdb/tree/master/importer#running-the-import-command) command.


#### `-datadir` string
Data storage path.

`default` = "$HOME/.influxdb/data"

#### `-waldir` string
WAL storage path.

`default` = "$HOME/.influxdb/wal"

#### `-out` string
Destination file to export to

`default` = "$HOME/.influxdb/export"

#### `-database` string (optional)
Database to export.

`default` = ""

#### `-retention` string (optional)
Retention policy to export.

`default` = ""

#### `-start` string (optional)
Optional. The time range to start at.

#### `-end` string (optional)
Optional. The time range to end at.

#### `-compress` bool (optional)
Compress the output.

`default` = false

#### Sample Commands

Export entire database and compress output:
```
influx_inspect export --compress
```

Export specific retention policy:
```
influx_inspect export --db mydb --rp autogen
```

##### Sample Data
This is a sample of what the output will look like.

```
# DDL
CREATE DATABASE MY_DB_NAME
CREATE RETENTION POLICY autogen ON MY_DB_NAME DURATION inf REPLICATION 1

# DML
# CONTEXT-DATABASE:MY_DB_NAME
# CONTEXT-RETENTION-POLICY:autogen
randset value=97.9296104805 1439856000000000000
randset value=25.3849066842 1439856100000000000
```

# Caveats

The system does not have access to the meta store when exporting TSM shards.  As such, it always creates the retention policy with infinite duration and replication factor of 1.
End users may want to change this prior to re-importing if they are importing to a cluster or want a different duration for retention.
