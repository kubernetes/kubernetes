OpenTSDB Input
============
InfluxDB supports both the telnet and HTTP OpenTSDB protocol. This means that InfluxDB can act as a drop-in replacement for your OpenTSDB system.

## Configuration
The OpenTSDB inputs allow the binding address, target database, and target retention policy within that database, to be set. If the database does not exist, it will be created automatically when the input is initialized. If you also decide to configure retention policy (without configuration the input will use the auto-created default retention policy), both the database and retention policy must already exist.

The write-consistency-level can also be set. If any write operations do not meet the configured consistency guarantees, an error will occur and the data will not be indexed. The default consistency-level is `ONE`.

The OpenTSDB input also performs internal batching of the points it receives, as batched writes to the database are more efficient. The default _batch size_ is 1000, _pending batch_ factor is 5, with a _batch timeout_ of 1 second. This means the input will write batches of maximum size 1000, but if a batch has not reached 1000 points within 1 second of the first point being added to a batch, it will emit that batch regardless of size. The pending batch factor controls how many batches can be in memory at once, allowing the input to transmit a batch, while still building other batches.
