## Upgrade etcd from 3.0 to 3.1

In the general case, upgrading from etcd 3.0 to 3.1 can be a zero-downtime, rolling upgrade:
 - one by one, stop the etcd v3.0 processes and replace them with etcd v3.1 processes
 - after running all v3.1 processes, new features in v3.1 are available to the cluster

Before [starting an upgrade](#upgrade-procedure), read through the rest of this guide to prepare.

### Upgrade checklists

#### Upgrade requirements

To upgrade an existing etcd deployment to 3.1, the running cluster must be 3.0 or greater. If it's before 3.0, please upgrade to [3.0](https://github.com/coreos/etcd/releases/tag/v3.0.16) before upgrading to 3.1.

Also, to ensure a smooth rolling upgrade, the running cluster must be healthy. Check the health of the cluster by using the `etcdctl endpoint health` command before proceeding.

#### Preparation

Before upgrading etcd, always test the services relying on etcd in a staging environment before deploying the upgrade to the production environment.

Before beginning, [backup the etcd data](../op-guide/maintenance.md#snapshot-backup). Should something go wrong with the upgrade, it is possible to use this backup to [downgrade](#downgrade) back to existing etcd version. Please note that the `snapshot` command only backs up the v3 data. For v2 data, see [backing up v2 datastore](../v2/admin_guide.md#backing-up-the-datastore).

#### Mixed versions

While upgrading, an etcd cluster supports mixed versions of etcd members, and operates with the protocol of the lowest common version. The cluster is only considered upgraded once all of its members are upgraded to version 3.1. Internally, etcd members negotiate with each other to determine the overall cluster version, which controls the reported version and the supported features.

#### Limitations

Note: If the cluster only has v3 data and no v2 data, it is not subject to this limitation.

If the cluster is serving a v2 data set larger than 50MB, each newly upgraded member may take up to two minutes to catch up with the existing cluster. Check the size of a recent snapshot to estimate the total data size. In other words, it is safest to wait for 2 minutes between upgrading each member.

For a much larger total data size, 100MB or more , this one-time process might take even more time. Administrators of very large etcd clusters of this magnitude can feel free to contact the [etcd team][etcd-contact] before upgrading, and we'll be happy to provide advice on the procedure.

#### Downgrade

If all members have been upgraded to v3.1, the cluster will be upgraded to v3.1, and downgrade from this completed state is **not possible**. If any single member is still v3.0, however, the cluster and its operations remains "v3.0", and it is possible from this mixed cluster state to return to using a v3.0 etcd binary on all members.

Please [backup the data directory](../op-guide/maintenance.md#snapshot-backup) of all etcd members to make downgrading the cluster possible even after it has been completely upgraded.

### Upgrade procedure

This example shows how to upgrade a 3-member v3.0 ectd cluster running on a local machine.

#### 1. Check upgrade requirements

Is the cluster healthy and running v3.0.x?

```
$ ETCDCTL_API=3 etcdctl endpoint health --endpoints=localhost:2379,localhost:22379,localhost:32379
localhost:2379 is healthy: successfully committed proposal: took = 6.600684ms
localhost:22379 is healthy: successfully committed proposal: took = 8.540064ms
localhost:32379 is healthy: successfully committed proposal: took = 8.763432ms

$ curl http://localhost:2379/version
{"etcdserver":"3.0.16","etcdcluster":"3.0.0"}
```

#### 2. Stop the existing etcd process

When each etcd process is stopped, expected errors will be logged by other cluster members. This is normal since a cluster member connection has been (temporarily) broken:

```
2017-01-17 09:34:18.352662 I | raft: raft.node: 1640829d9eea5cfb elected leader 1640829d9eea5cfb at term 5
2017-01-17 09:34:18.359630 W | etcdserver: failed to reach the peerURL(http://localhost:2380) of member fd32987dcd0511e0 (Get http://localhost:2380/version: dial tcp 127.0.0.1:2380: getsockopt: connection refused)
2017-01-17 09:34:18.359679 W | etcdserver: cannot get the version of member fd32987dcd0511e0 (Get http://localhost:2380/version: dial tcp 127.0.0.1:2380: getsockopt: connection refused)
2017-01-17 09:34:18.548116 W | rafthttp: lost the TCP streaming connection with peer fd32987dcd0511e0 (stream Message writer)
2017-01-17 09:34:19.147816 W | rafthttp: lost the TCP streaming connection with peer fd32987dcd0511e0 (stream MsgApp v2 writer)
2017-01-17 09:34:34.364907 W | etcdserver: failed to reach the peerURL(http://localhost:2380) of member fd32987dcd0511e0 (Get http://localhost:2380/version: dial tcp 127.0.0.1:2380: getsockopt: connection refused)
```

It's a good idea at this point to [backup the etcd data](../op-guide/maintenance.md#snapshot-backup) to provide a downgrade path should any problems occur:

```
$ etcdctl snapshot save backup.db
```

#### 3. Drop-in etcd v3.1 binary and start the new etcd process

The new v3.1 etcd will publish its information to the cluster:

```
2017-01-17 09:36:00.996590 I | etcdserver: published {Name:my-etcd-1 ClientURLs:[http://localhost:2379]} to cluster 46bc3ce73049e678
```

Verify that each member, and then the entire cluster, becomes healthy with the new v3.1 etcd binary:

```
$ ETCDCTL_API=3 /etcdctl endpoint health --endpoints=localhost:2379,localhost:22379,localhost:32379
localhost:22379 is healthy: successfully committed proposal: took = 5.540129ms
localhost:32379 is healthy: successfully committed proposal: took = 7.321671ms
localhost:2379 is healthy: successfully committed proposal: took = 10.629901ms
```

Upgraded members will log warnings like the following until the entire cluster is upgraded. This is expected and will cease after all etcd cluster members are upgraded to v3.1:

```
2017-01-17 09:36:38.406268 W | etcdserver: the local etcd version 3.0.16 is not up-to-date
2017-01-17 09:36:38.406295 W | etcdserver: member fd32987dcd0511e0 has a higher version 3.1.0
2017-01-17 09:36:42.407695 W | etcdserver: the local etcd version 3.0.16 is not up-to-date
2017-01-17 09:36:42.407730 W | etcdserver: member fd32987dcd0511e0 has a higher version 3.1.0
```

#### 4. Repeat step 2 to step 3 for all other members

#### 5. Finish

When all members are upgraded, the cluster will report upgrading to 3.1 successfully:

```
2017-01-17 09:37:03.100015 I | etcdserver: updating the cluster version from 3.0 to 3.1
2017-01-17 09:37:03.104263 N | etcdserver/membership: updated the cluster version from 3.0 to 3.1
2017-01-17 09:37:03.104374 I | etcdserver/api: enabled capabilities for version 3.1
```

```
$ ETCDCTL_API=3 /etcdctl endpoint health --endpoints=localhost:2379,localhost:22379,localhost:32379
localhost:2379 is healthy: successfully committed proposal: took = 2.312897ms
localhost:22379 is healthy: successfully committed proposal: took = 2.553476ms
localhost:32379 is healthy: successfully committed proposal: took = 2.516902ms
```

[etcd-contact]: https://groups.google.com/forum/#!forum/etcd-dev
