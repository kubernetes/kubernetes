# Interacting with etcd

Users mostly interact with etcd by putting or getting the value of a key. This section describes how to do that by using etcdctl, a command line tool for interacting with etcd server. The concepts described here should apply to the gRPC APIs or client library APIs.

By default, etcdctl talks to the etcd server with the v2 API for backward compatibility. For etcdctl to speak to etcd using the v3 API, the API version must be set to version 3 via the `ETCDCTL_API` environment variable.

``` bash
export ETCDCTL_API=3
```

## Write a key

Applications store keys into the etcd cluster by writing to keys. Every stored key is replicated to all etcd cluster members through the Raft protocol to achieve consistency and reliability.

Here is the command to set the value of key `foo` to `bar`:

``` bash
$ etcdctl put foo bar
OK
```

## Read keys

Applications can read values of keys from an etcd cluster. Queries may read a single key, or a range of keys. 

Suppose the etcd cluster has stored the following keys:

```
foo = bar
foo1 = bar1
foo3 = bar3
```

Here is the command to read the value of key `foo`:

```bash
$ etcdctl get foo
foo
bar
```

Here is the command to range over the keys from `foo` to `foo9`:

```bash
$ etcdctl get foo foo9
foo
bar
foo1
bar1
foo3
bar3
```

## Read past version of keys

Applications may want to read superseded versions of a key. For example, an application may wish to roll back to an old configuration by accessing an earlier version of a key. Alternatively, an application may want a consistent view over multiple keys through multiple requests by accessing key history.
Since every modification to the etcd cluster key-value store increments the global revision of an etcd cluster, an application can read superseded keys by providing an older etcd revision.

Suppose an etcd cluster already has the following keys:

``` bash
$ etcdctl put foo bar         # revision = 2
$ etcdctl put foo1 bar1       # revision = 3
$ etcdctl put foo bar_new     # revision = 4 
$ etcdctl put foo1 bar1_new   # revision = 5
```

Here are an example to access the past versions of keys:

```bash
$ etcdctl get foo foo9 # access the most recent versions of keys
foo
bar_new
foo1
bar1_new

$ etcdctl get --rev=4 foo foo9 # access the versions of keys at revision 4
foo
bar_new
foo1
bar1

$ etcdctl get --rev=3 foo foo9 # access the versions of keys at revision 3
foo
bar
foo1
bar1

$ etcdctl get --rev=2 foo foo9 # access the versions of keys at revision 2
foo
bar

$ etcdctl get --rev=1 foo foo9 # access the versions of keys at revision 1
```

## Delete keys

Applications can delete a key or a range of keys from an etcd cluster.

Here is the command to delete key `foo`:

```bash
$ etcdctl del foo
1 # one key is deleted
```

Here is the command to delete keys ranging from `foo` to `foo9`:

```bash
$ etcdctl del foo foo9
2 # two keys are deleted
```

## Watch key changes

Applications can watch on a key or a range of keys to monitor for any updates.

Here is the command to watch on key `foo`:

```bash
$ etcdctl watch foo 
# in another terminal: etcdctl put foo bar
foo
bar
```

Here is the command to watch on a range key from `foo` to `foo9`:

```bash
$ etcdctl watch foo foo9
# in another terminal: etcdctl put foo bar
foo
bar
# in another terminal: etcdctl put foo1 bar1
foo1
bar1
```

## Watch historical changes of keys

Applications may want to watch for historical changes of keys in etcd. For example, an application may wish to receive all the modifications of a key; if the application stays connected to etcd, then `watch` is good enough. However, if the application or etcd fails, a change may happen during the failure, and the application will not receive the update in real time. To guarantee the update is delivered, the application must be able to watch for historical changes to keys. To do this, an application can specify a historical revision on a watch, just like reading past version of keys.

Suppose we finished the following sequence of operations:

``` bash
etcdctl put foo bar         # revision = 2
etcdctl put foo1 bar1       # revision = 3
etcdctl put foo bar_new     # revision = 4 
etcdctl put foo1 bar1_new   # revision = 5
```

Here is an example to watch the historical changes:
```bash
# watch for changes on key `foo` since revision 2
$ etcdctl watch --rev=2 foo
PUT
foo
bar
PUT
foo
bar_new

# watch for changes on key `foo` since revision 3
$ etcdctl watch --rev=3 foo
PUT
foo
bar_new
```

## Compacted revisions

As we mentioned, etcd keeps revisions so that applications can read past versions of keys. However, to avoid accumulating an unbounded amount of history, it is important to compact past revisions. After compacting, etcd removes historical revisions, releasing resources for future use. All superseded data with revisions before the compacted revision will be unavailable.

Here is the command to compact the revisions:

```bash
$ etcdctl compact 5
compacted revision 5

# any revisions before the compacted one are not accessible
$ etcdctl get --rev=4 foo
Error:  rpc error: code = 11 desc = etcdserver: mvcc: required revision has been compacted
```

## Grant leases

Applications can grant leases for keys from an etcd cluster. When a key is attached to a lease, its lifetime is bound to the lease's lifetime which in turn is governed by a time-to-live (TTL). Each lease has a minimum time-to-live (TTL) value specified by the application at grant time. The lease's actual TTL value is at least the minimum TTL and is chosen by the etcd cluster. Once a lease's TTL elapses, the lease expires and all attached keys are deleted.

Here is the command to grant a lease:

```
# grant a lease with 10 second TTL
$ etcdctl lease grant 10
lease 32695410dcc0ca06 granted with TTL(10s)

# attach key foo to lease 32695410dcc0ca06
$ etcdctl put --lease=32695410dcc0ca06 foo bar
OK
```

## Revoke leases

Applications revoke leases by lease ID. Revoking a lease deletes all of its attached keys.

Suppose we finished the following sequence of operations:

```
$ etcdctl lease grant 10
lease 32695410dcc0ca06 granted with TTL(10s)
$ etcdctl put --lease=32695410dcc0ca06 foo bar
OK
```

Here is the command to revoke the same lease:

```
$ etcdctl lease revoke 32695410dcc0ca06
lease 32695410dcc0ca06 revoked

$ etcdctl get foo
# empty response since foo is deleted due to lease revocation
```

## Keep leases alive

Applications can keep a lease alive by refreshing its TTL so it does not expire.

Suppose we finished the following sequence of operations:

```
$ etcdctl lease grant 10
lease 32695410dcc0ca06 granted with TTL(10s)
```

Here is the command to keep the same lease alive:

```
$ etcdctl lease keep-alive 32695410dcc0ca0
lease 32695410dcc0ca0 keepalived with TTL(100)
lease 32695410dcc0ca0 keepalived with TTL(100)
lease 32695410dcc0ca0 keepalived with TTL(100)
...
```
