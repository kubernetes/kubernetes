# The collectd Input

The [collectd](https://collectd.org) input allows InfluxDB to accept data transmitted in collectd native format. This data is transmitted over UDP.

## A note on UDP/IP OS Buffer sizes

If you're running Linux or FreeBSD, please adjust your OS UDP buffer
size limit, [see here for more details.](../udp/README.md#a-note-on-udpip-os-buffer-sizes)

## Configuration

Each collectd input allows the binding address, target database, and target retention policy to be set. If the database does not exist, it will be created automatically when the input is initialized. If the retention policy is not configured, then the default retention policy for the database is used. However if the retention policy is set, the retention policy must be explicitly created. The input will not automatically create it.

Each collectd input also performs internal batching of the points it receives, as batched writes to the database are more efficient. The default batch size is 1000, pending batch factor is 5, with a batch timeout of 1 second. This means the input will write batches of maximum size 1000, but if a batch has not reached 1000 points within 1 second of the first point being added to a batch, it will emit that batch regardless of size. The pending batch factor controls how many batches can be in memory at once, allowing the input to transmit a batch, while still building other batches.

The path to the collectd types database file may also be set.

## Large UDP packets

Please note that UDP packets larger than the standard size of 1452 are dropped at the time of ingestion. Be sure to set `MaxPacketSize` to 1452 in the collectd configuration.

## Config Example

```
[[collectd]]
  enabled = true
  bind-address = ":25826" # the bind address
  database = "collectd" # Name of the database that will be written to
  retention-policy = ""
  batch-size = 5000 # will flush if this many points get buffered
  batch-pending = 10 # number of batches that may be pending in memory
  batch-timeout = "10s"
  read-buffer = 0 # UDP read buffer size, 0 means to use OS default
  typesdb = "/usr/share/collectd/types.db"
```
