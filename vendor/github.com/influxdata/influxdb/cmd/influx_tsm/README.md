# Converting b1 and bz1 shards to tsm1

`influx_tsm` is a tool for converting b1 and bz1 shards to tsm1
format. Converting shards to tsm1 format results in a very significant
reduction in disk usage, and significantly improved write-throughput,
when writing data into those shards.

Conversion can be controlled on a database-by-database basis. By
default a database is backed up before it is converted, allowing you
to roll back any changes. Because of the backup process, ensure the
host system has at least as much free disk space as the disk space
consumed by the _data_ directory of your InfluxDB system.

The tool automatically ignores tsm1 shards, and can be run
idempotently on any database.

Conversion is an offline process, and the InfluxDB system must be
stopped during conversion. However the conversion process reads and
writes shards directly on disk and should be fast.

## Steps

Follow these steps to perform a conversion.

* Identify the databases you wish to convert. You can convert one or more databases at a time. By default all databases are converted.
* Decide on parallel operation. By default the conversion operation peforms each operation in a serial manner. This minimizes load on the host system performing the conversion, but also takes the most time. If you wish to minimize the time conversion takes, enable parallel mode. Conversion will then perform as many operations as possible in parallel, but the process may place significant load on the host system (CPU, disk, and RAM, usage will all increase).
* Stop all write-traffic to your InfluxDB system.
* Restart the InfluxDB service and wait until all WAL data is flushed to disk -- this has completed when the system responds to queries. This is to ensure all data is present in shards.
* Stop the InfluxDB service. It should not be restarted until conversion is complete.
* Run conversion tool. Depending on the size of the data directory, this might be a lengthy operation. Consider running the conversion tool under a "screen" session to avoid any interruptions.
* Unless you ran the conversion tool as the same user as that which runs InfluxDB, then you may need to set the correct read-and-write permissions on the new tsm1 directories.
* Restart node and ensure data looks correct.
* If everything looks OK, you may then wish to remove or archive the backed-up databases.
* Restart write traffic.

## Example session

Below is an example session, showing a database being converted.

```
$ # Create a backup location that the `influxdb` user has full access to
$ mkdir -m 0777 /path/to/influxdb_backup
$ sudo -u influxdb influx_tsm -backup /path/to/influxdb_backup -parallel /var/lib/influxdb/data

b1 and bz1 shard conversion.
-----------------------------------
Data directory is:                  /var/lib/influxdb/data
Backup directory is:                /path/to/influxdb_backup
Databases specified:                all
Database backups enabled:           yes
Parallel mode enabled (GOMAXPROCS): yes (8)


Found 1 shards that will be converted.

Database        Retention       Path                                                    Engine  Size
_internal       monitor         /var/lib/influxdb/data/_internal/monitor/1           bz1     65536

These shards will be converted. Proceed? y/N: y
Conversion starting....
Backing up 1 databases...
2016/01/28 12:23:43.699266 Backup of databse '_internal' started
2016/01/28 12:23:43.699883 Backing up file /var/lib/influxdb/data/_internal/monitor/1
2016/01/28 12:23:43.700052 Database _internal backed up (851.776µs)
2016/01/28 12:23:43.700320 Starting conversion of shard: /var/lib/influxdb/data/_internal/monitor/1
2016/01/28 12:23:43.706276 Conversion of /var/lib/influxdb/data/_internal/monitor/1 successful (6.040148ms)

Summary statistics
========================================
Databases converted:                 1
Shards converted:                    1
TSM files created:                   1
Points read:                         369
Points written:                      369
NaN filtered:                        0
Inf filtered:                        0
Points without fields filtered:      0
Disk usage pre-conversion (bytes):   65536
Disk usage post-conversion (bytes):  11000
Reduction factor:                    83%
Bytes per TSM point:                 29.81
Total conversion time:               7.330443ms

$ # restart node, verify data
$ sudo rm -r /path/to/influxdb_backup
```

Note that the tool first lists the shards that will be converted,
before asking for confirmation. You can abort the conversion process
at this step if you just wish to see what would be converted, or if
the list of shards does not look correct.

__WARNING:__ If you run the `influx_tsm` tool as a user other than the
`influxdb` user (or the user that the InfluxDB process runs under),
please make sure to verify the shard permissions are correct prior to
starting InfluxDB. If needed, shard permissions can be corrected with
the `chown` command. For example:

```
sudo chown -R influxdb:influxdb /var/lib/influxdb
```

## Rolling back a conversion

After a successful backup (the message `Database XYZ backed up` was
logged), you have a duplicate of that database in the _backup_
directory you provided on the command line. If, when checking your
data after a successful conversion, you notice things missing or
something just isn't right, you can "undo" the conversion:

- Shut down your node (this is very important)
- Remove the database's directory from the influxdb `data` directory (default: `~/.influxdb/data/XYZ` for binary installations or `/var/lib/influxdb/data/XYZ` for packaged installations)
- Copy (to really make sure the shard is preserved) the database's directory from the backup directory you created into the `data` directory.

Using the same directories as above, and assuming a database named `stats`:

```
$ sudo rm -r /var/lib/influxdb/data/stats
$ sudo cp -r /path/to/influxdb_backup/stats /var/lib/influxdb/data/
$ # restart influxd node
```

#### How to avoid downtime when upgrading shards

*Identify non-`tsm1` shards*

Non-`tsm1` shards are files of the form: `data/<database>/<retention_policy>/<shard_id>`.

`tsm1` shards are files of the form: `data/<database>/<retention_policy>/<shard_id>/<file>.tsm`.

*Determine which `bz`/`bz1` shards are cold for writes*

Run the `SHOW SHARDS` query to see the start and end dates for shards.
If the date range for a shard does not span the current time then the shard is said to be cold for writes.
This means that no new points are expected to be added to the shard.
The shard whose date range spans now is said to be hot for writes.
You can only safely convert cold shards without stopping the InfluxDB process.

*Convert cold shards*

1. Copy each of the cold shards you'd like to convert to a new directory with the structure `/tmp/data/<database>/<retention_policy>/<shard_id>`.
2. Run the `influx_tsm` tool on the copied files:
```
influx_tsm -parallel /tmp/data/
```
3. Remove the existing cold `b1`/`bz1` shards from the production data directory.
4. Move the new `tsm1` shards into the original directory, overwriting the existing `b1`/`bz1` shards of the same name. Do this simultaneously with step 3 to avoid any query errors.
5. Wait an hour, a day, or a week (depending on your retention period) for any hot `b1`/`bz1` shards to become cold and repeat steps 1 through 4 on the newly cold shards.

> **Note:** Any points written to the cold shards after making a copy will be lost when the `tsm1` shard overwrites the existing cold shard.
Nothing in InfluxDB will prevent writes to cold shards, they are merely unexpected, not impossible.
It is your responsibility to prevent writes to cold shards to prevent data loss.
