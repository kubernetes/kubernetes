# Administration

## Data Directory

### Lifecycle

When first started, etcd stores its configuration into a data directory specified by the data-dir configuration parameter.
Configuration is stored in the write ahead log and includes: the local member ID, cluster ID, and initial cluster configuration.
The write ahead log and snapshot files are used during member operation and to recover after a restart.

Having a dedicated disk to store wal files can improve the throughput and stabilize the cluster. 
It is highly recommended to dedicate a wal disk and set `--wal-dir` to point to a directory on that device for a production cluster deployment.

If a member’s data directory is ever lost or corrupted then the user should [remove][remove-a-member] the etcd member from the cluster using `etcdctl` tool.

A user should avoid restarting an etcd member with a data directory from an out-of-date backup.
Using an out-of-date data directory can lead to inconsistency as the member had agreed to store information via raft then re-joins saying it needs that information again.
For maximum safety, if an etcd member suffers any sort of data corruption or loss, it must be removed from the cluster.
Once removed the member can be re-added with an empty data directory.

### Contents

The data directory has two sub-directories in it:

1. wal: write ahead log files are stored here. For details see the [wal package documentation][wal-pkg]
2. snap: log snapshots are stored here. For details see the [snap package documentation][snap-pkg]

If `--wal-dir` flag is set, etcd will write the write ahead log files to the specified directory instead of data directory.

## Cluster Management

### Lifecycle

If you are spinning up multiple clusters for testing it is recommended that you specify a unique initial-cluster-token for the different clusters.
This can protect you from cluster corruption in case of mis-configuration because two members started with different cluster tokens will refuse members from each other.

### Monitoring

It is important to monitor your production etcd cluster for healthy information and runtime metrics.

#### Health Monitoring

At lowest level, etcd exposes health information via HTTP at `/health` in JSON format. If it returns `{"health": "true"}`, then the cluster is healthy. Please note the `/health` endpoint is still an experimental one as in etcd 2.2.

```
$ curl -L http://127.0.0.1:2379/health

{"health": "true"}
```

You can also use etcdctl to check the cluster-wide health information. It will contact all the members of the cluster and collect the health information for you.

```
$./etcdctl cluster-health 
member 8211f1d0f64f3269 is healthy: got healthy result from http://127.0.0.1:12379
member 91bc3c398fb3c146 is healthy: got healthy result from http://127.0.0.1:22379
member fd422379fda50e48 is healthy: got healthy result from http://127.0.0.1:32379
cluster is healthy
```

#### Runtime Metrics

etcd uses [Prometheus][prometheus] for metrics reporting in the server. You can read more through the runtime metrics [doc][metrics].

### Debugging

Debugging a distributed system can be difficult. etcd provides several ways to make debug
easier.

#### Enabling Debug Logging

When you want to debug etcd without stopping it, you can enable debug logging at runtime.
etcd exposes logging configuration at `/config/local/log`.

```
$ curl http://127.0.0.1:2379/config/local/log -XPUT -d '{"Level":"DEBUG"}'
$ # debug logging enabled
$
$ curl http://127.0.0.1:2379/config/local/log -XPUT -d '{"Level":"INFO"}'
$ # debug logging disabled
```

#### Debugging Variables

Debug variables are exposed for real-time debugging purposes. Developers who are familiar with etcd can utilize these variables to debug unexpected behavior. etcd exposes debug variables via HTTP at `/debug/vars` in JSON format. The debug variables contains
`cmdline`, `file_descriptor_limit`, `memstats` and `raft.status`.

`cmdline` is the command line arguments passed into etcd.

`file_descriptor_limit` is the max number of file descriptors etcd can utilize.

`memstats` is explained in detail in the [Go runtime documentation][golang-memstats].

`raft.status` is useful when you want to debug low level raft issues if you are familiar with raft internals. In most cases, you do not need to check `raft.status`.

```json
{
"cmdline": ["./etcd"],
"file_descriptor_limit": 0,
"memstats": {"Alloc":4105744,"TotalAlloc":42337320,"Sys":12560632,"...":"..."},
"raft.status": {"id":"ce2a822cea30bfca","term":5,"vote":"ce2a822cea30bfca","commit":23509,"lead":"ce2a822cea30bfca","raftState":"StateLeader","progress":{"ce2a822cea30bfca":{"match":23509,"next":23510,"state":"ProgressStateProbe"}}}
}
```

### Optimal Cluster Size

The recommended etcd cluster size is 3, 5 or 7, which is decided by the fault tolerance requirement. A 7-member cluster can provide enough fault tolerance in most cases. While larger cluster provides better fault tolerance the write performance reduces since data needs to be replicated to more machines.

#### Fault Tolerance Table

It is recommended to have an odd number of members in a cluster. Having an odd cluster size doesn't change the number needed for majority, but you gain a higher tolerance for failure by adding the extra member. You can see this in practice when comparing even and odd sized clusters:

| Cluster Size | Majority   | Failure Tolerance |
|--------------|------------|-------------------|
| 1 | 1 | 0 |
| 2 | 2 | 0 |
| 3 | 2 | **1** |
| 4 | 3 | 1 |
| 5 | 3 | **2** |
| 6 | 4 | 2 |
| 7 | 4 | **3** |
| 8 | 5 | 3 |
| 9 | 5 | **4** |

As you can see, adding another member to bring the size of cluster up to an odd size is always worth it. During a network partition, an odd number of members also guarantees that there will almost always be a majority of the cluster that can continue to operate and be the source of truth when the partition ends.

#### Changing Cluster Size

After your cluster is up and running, adding or removing members is done via [runtime reconfiguration][runtime-reconfig], which allows the cluster to be modified without downtime. The `etcdctl` tool has `member list`, `member add` and `member remove` commands to complete this process.

### Member Migration

When there is a scheduled machine maintenance or retirement, you might want to migrate an etcd member to another machine without losing the data and changing the member ID.

The data directory contains all the data to recover a member to its point-in-time state. To migrate a member:

* Stop the member process.
* Copy the data directory of the now-idle member to the new machine.
* Update the peer URLs for the replaced member to reflect the new machine according to the [runtime reconfiguration instructions][update-a-member].
* Start etcd on the new machine, using the same configuration and the copy of the data directory.

This example will walk you through the process of migrating the infra1 member to a new machine:

|Name|Peer URL|
|------|--------------|
|infra0|10.0.1.10:2380|
|infra1|10.0.1.11:2380|
|infra2|10.0.1.12:2380|

```sh
$ export ETCDCTL_ENDPOINT=http://10.0.1.10:2379,http://10.0.1.11:2379,http://10.0.1.12:2379
```

```sh
$ etcdctl member list
84194f7c5edd8b37: name=infra0 peerURLs=http://10.0.1.10:2380 clientURLs=http://127.0.0.1:2379,http://10.0.1.10:2379
b4db3bf5e495e255: name=infra1 peerURLs=http://10.0.1.11:2380 clientURLs=http://127.0.0.1:2379,http://10.0.1.11:2379
bc1083c870280d44: name=infra2 peerURLs=http://10.0.1.12:2380 clientURLs=http://127.0.0.1:2379,http://10.0.1.12:2379
```

#### Stop the member etcd process

```sh
$ ssh 10.0.1.11
```

```sh
$ kill `pgrep etcd`
```

#### Copy the data directory of the now-idle member to the new machine

```
$ tar -cvzf infra1.etcd.tar.gz %data_dir%
```

```sh
$ scp infra1.etcd.tar.gz 10.0.1.13:~/
```

#### Update the peer URLs for that member to reflect the new machine

```sh
$ curl http://10.0.1.10:2379/v2/members/b4db3bf5e495e255 -XPUT \
-H "Content-Type: application/json" -d '{"peerURLs":["http://10.0.1.13:2380"]}'
```

Or use `etcdctl member update` command

```sh
$ etcdctl member update b4db3bf5e495e255 http://10.0.1.13:2380
```

#### Start etcd on the new machine, using the same configuration and the copy of the data directory

```sh
$ ssh 10.0.1.13
```

```sh
$ tar -xzvf infra1.etcd.tar.gz -C %data_dir%
```

```
etcd -name infra1 \
-listen-peer-urls http://10.0.1.13:2380 \
-listen-client-urls http://10.0.1.13:2379,http://127.0.0.1:2379 \
-advertise-client-urls http://10.0.1.13:2379,http://127.0.0.1:2379
```

### Disaster Recovery

etcd is designed to be resilient to machine failures. An etcd cluster can automatically recover from any number of temporary failures (for example, machine reboots), and a cluster of N members can tolerate up to _(N-1)/2_ permanent failures (where a member can no longer access the cluster, due to hardware failure or disk corruption). However, in extreme circumstances, a cluster might permanently lose enough members such that quorum is irrevocably lost. For example, if a three-node cluster suffered two simultaneous and unrecoverable machine failures, it would be normally impossible for the cluster to restore quorum and continue functioning.

To recover from such scenarios, etcd provides functionality to backup and restore the datastore and recreate the cluster without data loss.

#### Backing up the datastore

**Note:** Windows users must stop etcd before running the backup command.

The first step of the recovery is to backup the data directory and wal directory, if stored separately, on a functioning etcd node. To do this, use the `etcdctl backup` command, passing in the original data (and wal) directory used by etcd. For example:

```sh
    etcdctl backup \
      --data-dir %data_dir% \
      [--wal-dir %wal_dir%] \
      --backup-dir %backup_data_dir%
      [--backup-wal-dir %backup_wal_dir%]
```

This command will rewrite some of the metadata contained in the backup (specifically, the node ID and cluster ID), which means that the node will lose its former identity. In order to recreate a cluster from the backup, you will need to start a new, single-node cluster. The metadata is rewritten to prevent the new node from inadvertently being joined onto an existing cluster.

#### Restoring a backup

To restore a backup using the procedure created above, start etcd with the `-force-new-cluster` option and pointing to the backup directory. This will initialize a new, single-member cluster with the default advertised peer URLs, but preserve the entire contents of the etcd data store. Continuing from the previous example:

```sh
    etcd \
      -data-dir=%backup_data_dir% \
      [-wal-dir=%backup_wal_dir%] \
      -force-new-cluster \
      ...
```

Now etcd should be available on this node and serving the original datastore.

Once you have verified that etcd has started successfully, shut it down and move the data and wal, if stored separately, back to the previous location (you may wish to make another copy as well to be safe):

```sh
    pkill etcd
    rm -fr %data_dir%
    rm -fr %wal_dir%
    mv %backup_data_dir% %data_dir%
    mv %backup_wal_dir% %wal_dir%
    etcd \
      -data-dir=%data_dir% \
      [-wal-dir=%wal_dir%] \
      ...
```

#### Restoring the cluster

Now that the node is running successfully, [change its advertised peer URLs][update-a-member], as the `--force-new-cluster` option has set the peer URL to the default listening on localhost.

You can then add more nodes to the cluster and restore resiliency. See the [add a new member][add-a-member] guide for more details.

**Note:** If you are trying to restore your cluster using old failed etcd nodes, please make sure you have stopped old etcd instances and removed their old data directories specified by the data-dir configuration parameter.

### Client Request Timeout

etcd sets different timeouts for various types of client requests. The timeout value is not tunable now, which will be improved soon (https://github.com/coreos/etcd/issues/2038).

#### Get requests

Timeout is not set for get requests, because etcd serves the result locally in a non-blocking way.

**Note**: QuorumGet request is a different type, which is mentioned in the following sections.

#### Watch requests

Timeout is not set for watch requests. etcd will not stop a watch request until client cancels it, or the connection is broken.

#### Delete, Put, Post, QuorumGet requests

The default timeout is 5 seconds. It should be large enough to allow all key modifications if the majority of cluster is functioning.

If the request times out, it indicates two possibilities:

1. the server the request sent to was not functioning at that time.
2. the majority of the cluster is not functioning.

If timeout happens several times continuously, administrators should check status of cluster and resolve it as soon as possible.

### Best Practices

#### Maximum OS threads

By default, etcd uses the default configuration of the Go 1.4 runtime, which means that at most one operating system thread will be used to execute code simultaneously. (Note that this default behavior [has changed in Go 1.5][golang1.5-runtime]).

When using etcd in heavy-load scenarios on machines with multiple cores it will usually be desirable to increase the number of threads that etcd can utilize. To do this, simply set the environment variable GOMAXPROCS to the desired number when starting etcd. For more information on this variable, see the [Go runtime documentation][golang-runtime].

[add-a-member]: runtime-configuration.md#add-a-new-member
[golang1.5-runtime]: https://golang.org/doc/go1.5#runtime
[golang-memstats]: https://golang.org/pkg/runtime/#MemStats
[golang-runtime]: https://golang.org/pkg/runtime
[metrics]: metrics.md
[prometheus]: http://prometheus.io/
[remove-a-member]: runtime-configuration.md#remove-a-member
[runtime-reconfig]: runtime-configuration.md#cluster-reconfiguration-operations
[snap-pkg]: http://godoc.org/github.com/coreos/etcd/snap
[update-a-member]: runtime-configuration.md#update-a-member
[wal-pkg]: http://godoc.org/github.com/coreos/etcd/wal
