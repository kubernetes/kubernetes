# Upgrade etcd to 2.1

In the general case, upgrading from etcd 2.0 to 2.1 can be a zero-downtime, rolling upgrade:
 - one by one, stop the etcd v2.0 processes and replace them with etcd v2.1 processes
 - after you are running all v2.1 processes, new features in v2.1 are available to the cluster

Before [starting an upgrade](#upgrade-procedure), read through the rest of this guide to prepare.

## Upgrade Checklists

### Upgrade Requirements

To upgrade an existing etcd deployment to 2.1, you must be running 2.0. If you’re running a version of etcd before 2.0, you must upgrade to [2.0][v2.0] before upgrading to 2.1.

Also, to ensure a smooth rolling upgrade, your running cluster must be healthy. You can check the health of the cluster by using `etcdctl cluster-health` command. 

### Preparedness 

Before upgrading etcd, always test the services relying on etcd in a staging environment before deploying the upgrade to the production environment. 

You might also want to [backup your data directory][backup-datastore] for a potential [downgrade](#downgrade).

etcd 2.1 introduces a new [authentication][auth] feature, which is disabled by default. If your deployment depends on these, you may want to test the auth features before enabling them in production.

### Mixed Versions

While upgrading, an etcd cluster supports mixed versions of etcd members. The cluster is only considered upgraded once all its members are upgraded to 2.1.

Internally, etcd members negotiate with each other to determine the overall etcd cluster version, which controls the reported cluster version and the supported features. For example, if you are mid-upgrade, any 2.1 features (such as the the authentication feature mentioned above) won’t be available.

### Limitations

If you encounter any issues during the upgrade, you can attempt to restart the etcd process in trouble using a newer v2.1 binary to solve the problem. One known issue is that etcd v2.0.0 and v2.0.2 may panic during rolling upgrades due to an existing bug, which has been fixed since etcd v2.0.3.

It might take up to 2 minutes for the newly upgraded member to catch up with the existing cluster when the total data size is larger than 50MB (You can check the size of the existing snapshot to know about the rough data size). In other words, it is safest to wait for 2 minutes before upgrading the next member.

If you have even more data, this might take more time. If you have a data size larger than 100MB you should contact us before upgrading, so we can make sure the upgrades work smoothly.

### Downgrade

If all members have been upgraded to v2.1, the cluster will be upgraded to v2.1, and downgrade is **not possible**. If any member is still v2.0, the cluster will remain in v2.0, and you can go back to use v2.0 binary. 

Please [backup your data directory][backup-datastore] of all etcd members if you want to downgrade the cluster, even if it is upgraded.

### Upgrade Procedure

#### 1. Check upgrade requirements.

```
$ etcdctl cluster-health
cluster is healthy
member 6e3bd23ae5f1eae0 is healthy
member 924e2e83e93f2560 is healthy
member a8266ecf031671f3 is healthy

$ curl http://127.0.0.1:4001/version
etcd 2.0.x
```

#### 2. Stop the existing etcd process

You will see similar error logging from other etcd processes in your cluster. This is normal, since you just shut down a member.

```
2015/06/23 15:45:09 sender: error posting to 6e3bd23ae5f1eae0: dial tcp 127.0.0.1:7002: connection refused
2015/06/23 15:45:09 sender: the connection with 6e3bd23ae5f1eae0 became inactive
2015/06/23 15:45:11 rafthttp: encountered error writing to server log stream: write tcp 127.0.0.1:53783: broken pipe
2015/06/23 15:45:11 rafthttp: server streaming to 6e3bd23ae5f1eae0 at term 2 has been stopped
2015/06/23 15:45:11 stream: error sending message: stopped
2015/06/23 15:45:11 stream: stopping the stream server...
```

You could [backup your data directory][backup-datastore] for data safety.

```
$ etcdctl backup \
      --data-dir /var/lib/etcd \
      --backup-dir /tmp/etcd_backup
```

#### 3. Drop-in etcd v2.1 binary and start the new etcd process

You will see the etcd publish its information to the cluster.

```
2015/06/23 15:45:39 etcdserver: published {Name:infra2 ClientURLs:[http://localhost:4002]} to cluster e9c7614f68f35fb2
```

You could verify the cluster becomes healthy.

```
$ etcdctl cluster-health
cluster is healthy
member 6e3bd23ae5f1eae0 is healthy
member 924e2e83e93f2560 is healthy
member a8266ecf031671f3 is healthy
```

#### 4. Repeat step 2 to step 3 for all other members 

#### 5. Finish

When all members are upgraded, you will see the cluster is upgraded to 2.1 successfully:

```
2015/06/23 15:46:35 etcdserver: updated the cluster version from 2.0.0 to 2.1.0
```

```
$ curl http://127.0.0.1:4001/version
{"etcdserver":"2.1.x","etcdcluster":"2.1.0"}
```

[auth]: auth_api.md
[backup-datastore]: admin_guide.md#backing-up-the-datastore
[v2.0]: https://github.com/coreos/etcd/releases/tag/v2.0.13
