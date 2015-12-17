Shard Precreation
============

During normal operation when InfluxDB receives time-series data, it writes the data to files known as _shards_. Each shard only contains data for a specific range of time. Therefore, before data can be accepted by the system, the shards must exist and InfluxDB always checks that the required shards exist for every incoming data point. If the required shards do not exist, InfluxDB will create those shards. Because this requires a cluster to reach consensus, the process is not instantaneous and can temporarily impact write-throughput.

Since almost all time-series data is written sequentially in time, the system has an excellent idea of the timestamps of future data. Shard precreation takes advantage of this fact by creating required shards ahead of time, thereby ensuring the required shards exist by the time new time-series data actually arrives. Write-throughput is therefore not affected when data is first received for a range of time that would normally trigger shard creation.

Note that the shard-existence check must remain in place in the code, even with shard precreation. This is because while most data is written sequentially in time, this is not always the case. Data may be written with timestamps in the past, or farther in the future than shard precreation handles.

## Configuration
Shard precreation can be disabled if necessary, though this is not recommended. If it is disabled, then shards will be only be created when explicitly needed.

The interval between runs of the shard precreation service, as well as the time-in-advance the shards are created, are also configurable. The defaults should work for most deployments.
