# Maintenance

## Overview

An etcd cluster needs periodic maintenance to remain reliable. Depending on an etcd application's needs, this maintenance can usually be automated and performed without downtime or significantly degraded performance.

All etcd maintenance manages storage resources consumed by the etcd keyspace. Failure to adequately control the keyspace size is guarded by storage space quotas; if an etcd member runs low on space, a quota will trigger cluster-wide alarms which will put the system into a limited-operation maintenance mode. To avoid running out of space for writes to the keyspace, the etcd keyspace history must be compacted. Storage space itself may be reclaimed by defragmenting etcd members. Finally, periodic snapshot backups of etcd member state makes it possible to recover any unintended logical data loss or corruption caused by operational error.

## History compaction

Since etcd keeps an exact history of its keyspace, this history should be periodically compacted to avoid performance degradation and eventual storage space exhaustion. Compacting the keyspace history drops all information about keys superseded prior to a given keyspace revision. The space used by these keys then becomes available for additional writes to the keyspace.

The keyspace can be compacted automatically with `etcd`'s time windowed history retention policy, or manually with `etcdctl`. The `etcdctl` method provides fine-grained control over the compacting process whereas automatic compacting fits applications that only need key history for some length of time.

`etcd` can be set to automatically compact the keyspace with the `--auto-compaction` option with a period of hours:

```sh
# keep one hour of history
$ etcd --auto-compaction-retention=1
```

An `etcdctl` initiated compaction works as follows:

```sh
# compact up to revision 3
$ etcdctl compact 3

```

Revisions prior to the compaction revision become inaccessible:

```sh
$ etcdctl get --rev=2 somekey
Error:  rpc error: code = 11 desc = etcdserver: mvcc: required revision has been compacted
```

## Defragmentation

After compacting the keyspace, the backend database may exhibit internal fragmentation. Any internal fragmentation is space that is free to use by the backend but still consumes storage space. The process of defragmentation releases this storage space back to the file system. Defragmentation is issued on a per-member so that cluster-wide latency spikes may be avoided.

Compacting old revisions internally fragments `etcd` by leaving gaps in backend database. Fragmented space is available for use by `etcd` but unavailable to the host filesystem.

To defragment an etcd member, use the `etcdctl defrag` command:

```sh
$ etcdctl defrag
Finished defragmenting etcd member[127.0.0.1:2379]
```

## Space quota

The space quota in `etcd` ensures the cluster operates in a reliable fashion. Without a space quota, `etcd` may suffer from poor performance if the keyspace grows excessively large, or it may simply run out of storage space, leading to unpredictable cluster behavior. If the keyspace's backend database for any member exceeds the space quota, `etcd` raises a cluster-wide alarm that puts the cluster into a maintenance mode which only accepts key reads and deletes. After freeing enough space in the keyspace, the alarm can be disarmed and the cluster will resume normal operation.

By default, `etcd` sets a conservative space quota suitable for most applications, but it may be configured on the command line, in bytes:

```sh
# set a very small 16MB quota
$ etcd --quota-backend-bytes=16777216
```

The space quota can be triggered with a loop:

```sh
# fill keyspace
$ while [ 1 ]; do dd if=/dev/urandom bs=1024 count=1024  | etcdctl put key  || break; done
...
Error:  rpc error: code = 8 desc = etcdserver: mvcc: database space exceeded
# confirm quota space is exceeded
$ etcdctl --write-out=table endpoint status
+----------------+------------------+-----------+---------+-----------+-----------+------------+
|    ENDPOINT    |        ID        |  VERSION  | DB SIZE | IS LEADER | RAFT TERM | RAFT INDEX |
+----------------+------------------+-----------+---------+-----------+-----------+------------+
| 127.0.0.1:2379 | bf9071f4639c75cc | 2.3.0+git | 18 MB   | true      |         2 |       3332 |
+----------------+------------------+-----------+---------+-----------+-----------+------------+
# confirm alarm is raised
$ etcdctl alarm list
memberID:13803658152347727308 alarm:NOSPACE 
```

Removing excessive keyspace data will put the cluster back within the quota limits so the alarm can be disarmed:

```sh
# get current revision
$ etcdctl --endpoints=:2379 endpoint status
[{"Endpoint":"127.0.0.1:2379","Status":{"header":{"cluster_id":8925027824743593106,"member_id":13803658152347727308,"revision":1516,"raft_term":2},"version":"2.3.0+git","dbSize":17973248,"leader":13803658152347727308,"raftIndex":6359,"raftTerm":2}}]
# compact away all old revisions
$ etdctl compact 1516
compacted revision 1516
# defragment away excessive space
$ etcdctl defrag
Finished defragmenting etcd member[127.0.0.1:2379]
# disarm alarm
$ etcdctl alarm disarm
memberID:13803658152347727308 alarm:NOSPACE 
# test puts are allowed again
$ etdctl put newkey 123
OK
```

## Snapshot backup

Snapshotting the `etcd` cluster on a regular basis serves as a durable backup for an etcd keyspace. By taking periodic snapshots of an etcd member's backend database, an `etcd` cluster can be recovered to a point in time with a known good state.

A snapshot is taken with `etcdctl`:

```sh
$ etcdctl snapshot save backup.db
$ etcdctl --write-out=table snapshot status backup.db
+----------+----------+------------+------------+
|   HASH   | REVISION | TOTAL KEYS | TOTAL SIZE |
+----------+----------+------------+------------+
| fe01cf57 |       10 |          7 | 2.1 MB     |
+----------+----------+------------+------------+

```
