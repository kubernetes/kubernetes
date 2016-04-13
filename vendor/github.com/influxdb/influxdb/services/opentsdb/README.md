openTSDB Input
============
InfluxDB supports both the telnet and HTTP openTSDB protocol. This means that InfluxDB can act as a drop-in replacement for your openTSDB system.

## Configuration
The openTSDB input allows the binding address, target database, and target retention policy within that database, to be set. If the database does not exist, it will be created automatically when the input is initialized. If you also decide to configure retention policy (without configuration the input will use the auto-created default retention policy), both the database and retention policy must already exist.

The write-consistency-level can also be set. If any write operations do not meet the configured consistency guarantees, an error will occur and the data will not be indexed. The default consistency-level is `ONE`.
