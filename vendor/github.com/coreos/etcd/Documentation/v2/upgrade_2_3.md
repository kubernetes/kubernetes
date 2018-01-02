## Upgrade etcd from 2.2 to 2.3

In the general case, upgrading from etcd 2.2 to 2.3 can be a zero-downtime, rolling upgrade:
 - one by one, stop the etcd v2.2 processes and replace them with etcd v2.3 processes
 - after running all v2.3 processes, new features in v2.3 are available to the cluster

Before [starting an upgrade](#upgrade-procedure), read through the rest of this guide to prepare.

### Upgrade Checklists

#### Upgrade Requirements

To upgrade an existing etcd deployment to 2.3, the running cluster must be 2.2 or greater. If it's before 2.2, please upgrade to [2.2](https://github.com/coreos/etcd/releases/tag/v2.2.0) before upgrading to 2.3.

Also, to ensure a smooth rolling upgrade, the running cluster must be healthy. You can check the health of the cluster by using the `etcdctl cluster-health` command.

#### Preparation

Before upgrading etcd, always test the services relying on etcd in a staging environment before deploying the upgrade to the production environment.

Before beginning,  [backup the etcd data directory](admin_guide.md#backing-up-the-datastore). Should something go wrong with the upgrade, it is possible to use this backup to[downgrade](#downgrade) back to existing etcd version.

#### Mixed Versions

While upgrading, an etcd cluster supports mixed versions of etcd members, and operates with the protocol of the lowest common version. The cluster is only considered upgraded once all of its members are upgraded to version 2.3. Internally, etcd members negotiate with each other to determine the overall cluster version, which controls the reported version and the supported features.

#### Limitations

It might take up to 2 minutes for the newly upgraded member to catch up with the existing cluster when the total data size is larger than 50MB. Check the size of a recent  snapshot to estimate  the total data size. In other words, it is safest to wait for 2 minutes between upgrading each member.

For a much larger total data size, 100MB or more , this one-time process might take even more time. Administrators of very large etcd clusters of this magnitude can feel free to contact the [etcd team][etcd-contact] before upgrading, and we’ll be happy to provide advice on the procedure.

#### Downgrade

If all members have been upgraded to v2.3, the cluster will be upgraded to v2.3, and downgrade from this completed state is **not possible**. If any single member is still v2.2, however, the cluster and its operations remains “v2.2”, and it is possible from this mixed cluster state to return to using a v2.2 etcd binary on all members.

Please [backup the data directory](admin_guide.md#backing-up-the-datastore) of all etcd members to make downgrading the cluster possible even after it has been completely upgraded.

### Upgrade Procedure


This example details the  upgrade of a three-member v2.2 ectd cluster running on a local machine.

#### 1. Check upgrade requirements.

Is the the cluster healthy and running v.2.2.x?

```
$ etcdctl cluster-health
member 6e3bd23ae5f1eae0 is healthy: got healthy result from http://localhost:22379
member 924e2e83e93f2560 is healthy: got healthy result from http://localhost:32379
member a8266ecf031671f3 is healthy: got healthy result from http://localhost:12379
cluster is healthy

$ curl http://localhost:4001/version
{"etcdserver":"2.2.x","etcdcluster":"2.2.0"}
```

#### 2. Stop the existing etcd process

When each etcd process is stopped, expected errors will be logged by other cluster members. This is normal since a cluster member connection has been (temporarily) broken:

```
2016-03-11 09:50:49.860319 E | rafthttp: failed to read 8211f1d0f64f3269 on stream Message (unexpected EOF)
2016-03-11 09:50:49.860335 I | rafthttp: the connection with 8211f1d0f64f3269 became inactive
2016-03-11 09:50:51.023804 W | etcdserver: failed to reach the peerURL(http://127.0.0.1:12380) of member 8211f1d0f64f3269 (Get http://127.0.0.1:12380/version: dial tcp 127.0.0.1:12380: getsockopt: connection refused)
2016-03-11 09:50:51.023821 W | etcdserver: cannot get the version of member 8211f1d0f64f3269 (Get http://127.0.0.1:12380/version: dial tcp 127.0.0.1:12380: getsockopt: connection refused)
```

It’s a good idea at this point to  [backup the etcd data directory](https://github.com/coreos/etcd/blob/7f7e2cc79d9c5c342a6eb1e48c386b0223cf934e/Documentation/admin_guide.md#backing-up-the-datastore) to provide a downgrade path should any problems occur:

```
$ etcdctl backup \
      --data-dir /var/lib/etcd \
      --backup-dir /tmp/etcd_backup
```

#### 3. Drop-in etcd v2.3 binary and start the new etcd process

The new v2.3 etcd will publish its information to the cluster:

```
09:58:25.938673 I | etcdserver: published {Name:infra1 ClientURLs:[http://localhost:12379]} to cluster 524400597fb1d5f6
```

Verify that each member, and then the entire cluster, becomes healthy with the new v2.3 etcd binary:

```
$ etcdctl cluster-health
member 6e3bd23ae5f1eae0 is healthy: got healthy result from http://localhost:22379
member 924e2e83e93f2560 is healthy: got healthy result from http://localhost:32379
member a8266ecf031671f3 is healthy: got healthy result from http://localhost:12379
cluster is healthy
```


Upgraded members will log warnings like the following until the entire cluster is upgraded. This is expected and will cease after all etcd cluster members are upgraded to v2.3:

```
2016-03-11 09:58:26.851837 W | etcdserver: the local etcd version 2.2.0 is not up-to-date
2016-03-11 09:58:26.851854 W | etcdserver: member c02c70ede158499f has a higher version 2.3.0
```

#### 4. Repeat step 2 to step 3 for all other members

#### 5. Finish

When all members are upgraded, the cluster will report  upgrading to 2.3 successfully:

```
2016-03-11 10:03:01.583392 N | etcdserver: updated the cluster version from 2.2 to 2.3
```

```
$ curl http://127.0.0.1:4001/version
{"etcdserver":"2.3.x","etcdcluster":"2.3.0"}
```


[etcd-contact]: https://coreos.com/etcd/?

