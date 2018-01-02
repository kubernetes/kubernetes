# Upgrade etcd from 2.1 to 2.2

In the general case, upgrading from etcd 2.1 to 2.2 can be a zero-downtime, rolling upgrade:

 - one by one, stop the etcd v2.1 processes and replace them with etcd v2.2 processes
 - after you are running all v2.2 processes, new features in v2.2 are available to the cluster

Before [starting an upgrade](#upgrade-procedure), read through the rest of this guide to prepare.

## Upgrade Checklists

### Upgrade Requirement

To upgrade an existing etcd deployment to 2.2, you must be running 2.1. If you’re running a version of etcd before 2.1, you must upgrade to [2.1][v2.1] before upgrading to 2.2.

Also, to ensure a smooth rolling upgrade, your running cluster must be healthy. You can check the health of the cluster by using `etcdctl cluster-health` command. 

### Preparedness 

Before upgrading etcd, always test the services relying on etcd in a staging environment before deploying the upgrade to the production environment. 

You might also want to [backup the data directory][backup-datastore] for a potential [downgrade].

### Mixed Versions

While upgrading, an etcd cluster supports mixed versions of etcd members. The cluster is only considered upgraded once all its members are upgraded to 2.2.

Internally, etcd members negotiate with each other to determine the overall etcd cluster version, which controls the reported cluster version and the supported features.

### Limitations

If you have a data size larger than 100MB you should contact us before upgrading, so we can make sure the upgrades work smoothly.

Every etcd 2.2 member will do health checking across the cluster periodically. etcd 2.1 member does not support health checking. During the upgrade, etcd 2.2 member will log warning about the unhealthy state of etcd 2.1 member. You can ignore the warning. 

### Downgrade

If all members have been upgraded to v2.2, the cluster will be upgraded to v2.2, and downgrade is **not possible**. If any member is still v2.1, the cluster will remain in v2.1, and you can go back to use v2.1 binary. 

Please [backup the data directory][backup-datastore] of all etcd members if you want to downgrade the cluster, even if it is upgraded.

### Upgrade Procedure

In the example, we upgrade a three member v2.1 cluster running on local machine.

#### 1. Check upgrade requirements.

```
$ etcdctl cluster-health
member 6e3bd23ae5f1eae0 is healthy: got healthy result from http://localhost:22379
member 924e2e83e93f2560 is healthy: got healthy result from http://localhost:32379
member a8266ecf031671f3 is healthy: got healthy result from http://localhost:12379
cluster is healthy

$ curl http://localhost:4001/version
{"etcdserver":"2.1.x","etcdcluster":"2.1.0"}
```

#### 2. Stop the existing etcd process

You will see similar error logging from other etcd processes in your cluster. This is normal, since you just shut down a member and the connection is broken.

```
2015/09/2 09:48:35 etcdserver: failed to reach the peerURL(http://localhost:12380) of member a8266ecf031671f3 (Get http://localhost:12380/version: dial tcp [::1]:12380: getsockopt: connection refused)
2015/09/2 09:48:35 etcdserver: cannot get the version of member a8266ecf031671f3 (Get http://localhost:12380/version: dial tcp [::1]:12380: getsockopt: connection refused)
2015/09/2 09:48:35 rafthttp: failed to write a8266ecf031671f3 on stream Message (write tcp 127.0.0.1:32380->127.0.0.1:64394: write: broken pipe)
2015/09/2 09:48:35 rafthttp: failed to write a8266ecf031671f3 on pipeline (dial tcp [::1]:12380: getsockopt: connection refused)
2015/09/2 09:48:40 etcdserver: failed to reach the peerURL(http://localhost:7001) of member a8266ecf031671f3 (Get http://localhost:7001/version: dial tcp [::1]:12380: getsockopt: connection refused)
2015/09/2 09:48:40 etcdserver: cannot get the version of member a8266ecf031671f3 (Get http://localhost:12380/version: dial tcp [::1]:12380: getsockopt: connection refused)
2015/09/2 09:48:40 rafthttp: failed to heartbeat a8266ecf031671f3 on stream MsgApp v2 (write tcp 127.0.0.1:32380->127.0.0.1:64393: write: broken pipe)
```

You will see logging output like this from ungraded member due to a mixed version cluster. You can ignore this while upgrading.

```
2015/09/2 09:48:45 etcdserver: the etcd version 2.1.2+git is not up-to-date
2015/09/2 09:48:45 etcdserver: member a8266ecf031671f3 has a higher version &{2.2.0-rc.0+git 2.1.0}
```

You will also see logging output like this from the newly upgraded member, since etcd 2.1 member does not support health checking. You can ignore this while upgrading.

```
2015-09-02 09:55:42.691384 W | rafthttp: the connection to peer 6e3bd23ae5f1eae0 is unhealthy
2015-09-02 09:55:42.705626 W | rafthttp: the connection to peer 924e2e83e93f2560 is unhealthy

```

[Backup your data directory][backup-datastore] for data safety.

```
$ etcdctl backup \
      --data-dir /var/lib/etcd \
      --backup-dir /tmp/etcd_backup
```

#### 3. Drop-in etcd v2.2 binary and start the new etcd process

Now, you can start the etcd v2.2 binary with the previous configuration.
You will see the etcd start and publish its information to the cluster.

```
2015-09-02 09:56:46.117609 I | etcdserver: published {Name:infra2 ClientURLs:[http://localhost:22380]} to cluster e9c7614f68f35fb2
```

You could verify the cluster becomes healthy.

```
$ etcdctl cluster-health
member 6e3bd23ae5f1eae0 is healthy: got healthy result from http://localhost:22379
member 924e2e83e93f2560 is healthy: got healthy result from http://localhost:32379
member a8266ecf031671f3 is healthy: got healthy result from http://localhost:12379
cluster is healthy
```

#### 4. Repeat step 2 to step 3 for all other members 

#### 5. Finish

When all members are upgraded, you will see the cluster is upgraded to 2.2 successfully:

```
2015-09-02 09:56:54.896848 N | etcdserver: updated the cluster version from 2.1 to 2.2
```

```
$ curl http://127.0.0.1:4001/version
{"etcdserver":"2.2.x","etcdcluster":"2.2.0"}
```

[backup-datastore]: admin_guide.md#backing-up-the-datastore
[downgrade]: #downgrade
[v2.1]: https://github.com/coreos/etcd/releases/tag/v2.1.2
