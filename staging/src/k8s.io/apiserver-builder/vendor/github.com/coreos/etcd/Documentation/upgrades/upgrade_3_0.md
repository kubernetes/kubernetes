## Upgrade etcd from 2.3 to 3.0

In the general case, upgrading from etcd 2.3 to 3.0 can be a zero-downtime, rolling upgrade:
 - one by one, stop the etcd v2.3 processes and replace them with etcd v3.0 processes
 - after running all v3.0 processes, new features in v3.0 are available to the cluster

Before [starting an upgrade](#upgrade-procedure), read through the rest of this guide to prepare.

### Upgrade Checklists

#### Upgrade Requirements

To upgrade an existing etcd deployment to 3.0, the running cluster must be 2.3 or greater. If it's before 2.3, please upgrade to [2.3](https://github.com/coreos/etcd/releases/tag/v2.3.0) before upgrading to 3.0.

Also, to ensure a smooth rolling upgrade, the running cluster must be healthy. You can check the health of the cluster by using the `etcdctl cluster-health` command.

#### Preparation

Before upgrading etcd, always test the services relying on etcd in a staging environment before deploying the upgrade to the production environment.

Before beginning,  [backup the etcd data directory](../v2/admin_guide.md#backing-up-the-datastore). Should something go wrong with the upgrade, it is possible to use this backup to [downgrade](#downgrade) back to existing etcd version.

#### Mixed Versions

While upgrading, an etcd cluster supports mixed versions of etcd members, and operates with the protocol of the lowest common version. The cluster is only considered upgraded once all of its members are upgraded to version 3.0. Internally, etcd members negotiate with each other to determine the overall cluster version, which controls the reported version and the supported features.

#### Limitations

It might take up to 2 minutes for the newly upgraded member to catch up with the existing cluster when the total data size is larger than 50MB. Check the size of a recent  snapshot to estimate  the total data size. In other words, it is safest to wait for 2 minutes between upgrading each member.

For a much larger total data size, 100MB or more , this one-time process might take even more time. Administrators of very large etcd clusters of this magnitude can feel free to contact the [etcd team][etcd-contact] before upgrading, and we’ll be happy to provide advice on the procedure.

#### Downgrade

If all members have been upgraded to v3.0, the cluster will be upgraded to v3.0, and downgrade from this completed state is **not possible**. If any single member is still v2.3, however, the cluster and its operations remains “v2.3”, and it is possible from this mixed cluster state to return to using a v2.3 etcd binary on all members.

Please [backup the data directory](../v2/admin_guide.md#backing-up-the-datastore) of all etcd members to make downgrading the cluster possible even after it has been completely upgraded.

### Upgrade Procedure

This example details the  upgrade of a three-member v2.3 ectd cluster running on a local machine.

#### 1. Check upgrade requirements.

Is the the cluster healthy and running v.2.3.x?

```
$ etcdctl cluster-health
member 6e3bd23ae5f1eae0 is healthy: got healthy result from http://localhost:22379
member 924e2e83e93f2560 is healthy: got healthy result from http://localhost:32379
member 8211f1d0f64f3269 is healthy: got healthy result from http://localhost:12379
cluster is healthy

$ curl http://localhost:2379/version
{"etcdserver":"2.3.x","etcdcluster":"2.3.0"}
```

#### 2. Stop the existing etcd process

When each etcd process is stopped, expected errors will be logged by other cluster members. This is normal since a cluster member connection has been (temporarily) broken:

```
2016-06-27 15:21:48.624124 E | rafthttp: failed to dial 8211f1d0f64f3269 on stream Message (dial tcp 127.0.0.1:12380: getsockopt: connection refused)
2016-06-27 15:21:48.624175 I | rafthttp: the connection with 8211f1d0f64f3269 became inactive
```

It’s a good idea at this point to  [backup the etcd data directory](../v2/admin_guide.md#backing-up-the-datastore) to provide a downgrade path should any problems occur:

```
$ etcdctl backup \
      --data-dir /var/lib/etcd \
      --backup-dir /tmp/etcd_backup
```

#### 3. Drop-in etcd v3.0 binary and start the new etcd process

The new v3.0 etcd will publish its information to the cluster:

```
09:58:25.938673 I | etcdserver: published {Name:infra1 ClientURLs:[http://localhost:12379]} to cluster 524400597fb1d5f6
```

Verify that each member, and then the entire cluster, becomes healthy with the new v3.0 etcd binary:

```
$ etcdctl cluster-health
member 6e3bd23ae5f1eae0 is healthy: got healthy result from http://localhost:22379
member 924e2e83e93f2560 is healthy: got healthy result from http://localhost:32379
member 8211f1d0f64f3269 is healthy: got healthy result from http://localhost:12379
cluster is healthy
```


Upgraded members will log warnings like the following until the entire cluster is upgraded. This is expected and will cease after all etcd cluster members are upgraded to v3.0:

```
2016-06-27 15:22:05.679644 W | etcdserver: the local etcd version 2.3.7 is not up-to-date
2016-06-27 15:22:05.679660 W | etcdserver: member 8211f1d0f64f3269 has a higher version 3.0.0
```

#### 4. Repeat step 2 to step 3 for all other members

#### 5. Finish

When all members are upgraded, the cluster will report  upgrading to 3.0 successfully:

```
2016-06-27 15:22:19.873751 N | membership: updated the cluster version from 2.3 to 3.0
2016-06-27 15:22:19.914574 I | api: enabled capabilities for version 3.0.0
```

```
$ ETCDCTL_API=3 etcdctl endpoint health
127.0.0.1:12379 is healthy: successfully committed proposal: took = 18.440155ms
127.0.0.1:32379 is healthy: successfully committed proposal: took = 13.651368ms
127.0.0.1:22379 is healthy: successfully committed proposal: took = 18.513301ms
```

[etcd-contact]: https://groups.google.com/forum/#!forum/etcd-dev
