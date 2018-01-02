# Interacting with etcd

Users mostly interact with etcd by putting or getting the value of a key. This section describes how to do that by using etcdctl, a command line tool for interacting with etcd server. The concepts described here should apply to the gRPC APIs or client library APIs.

By default, etcdctl talks to the etcd server with the v2 API for backward compatibility. For etcdctl to speak to etcd using the v3 API, the API version must be set to version 3 via the `ETCDCTL_API` environment variable.

```bash
export ETCDCTL_API=3
```

## Find versions

etcdctl version and Server API version can be useful in finding the appropriate commands to be used for performing various opertions on etcd.

Here is the command to find the versions:

```bash
$ etcdctl version
etcdctl version: 3.1.0-alpha.0+git
API version: 3.1
```

## Write a key

Applications store keys into the etcd cluster by writing to keys. Every stored key is replicated to all etcd cluster members through the Raft protocol to achieve consistency and reliability.

Here is the command to set the value of key `foo` to `bar`:

```bash
$ etcdctl put foo bar
OK
```

Also a key can be set for a specified interval of time by attaching lease to it.

Here is the command to set the value of key `foo1` to `bar1` for 10s.

```bash
$ etcdctl put foo1 bar1 --lease=1234abcd
OK
```

Note: The lease id `1234abcd` in the above command refers to id returned on creating the lease of 10s. This id can then be attached to the key.

## Read keys

Applications can read values of keys from an etcd cluster. Queries may read a single key, or a range of keys.

Suppose the etcd cluster has stored the following keys:

```bash
foo = bar
foo1 = bar1
foo2 = bar2
foo3 = bar3
```

Here is the command to read the value of key `foo`:

```bash
$ etcdctl get foo
foo
bar
```

Here is the command to read the value of key `foo` in hex format:

```bash
$ etcdctl get foo --hex
\x66\x6f\x6f          # Key
\x62\x61\x72          # Value
```

Here is the command to read only the value of key `foo`:

```bash
$ etcdctl get foo --print-value-only
bar
```

Here is the command to range over the keys from `foo` to `foo3`:

```bash
$ etcdctl get foo foo3
foo
bar
foo1
bar1
foo2
bar2
```

Note that `foo3` is excluded since the range is over the half-open interval `[foo, foo3)`, excluding `foo3`.

Here is the command to range over all keys prefixed with `foo`:

```bash
$ etcdctl get --prefix foo
foo
bar
foo1
bar1
foo2
bar2
foo3
bar3
```

Here is the command to range over all keys prefixed with `foo`, limiting the number of results to 2:

```bash
$ etcdctl get --prefix --limit=2 foo
foo
bar
foo1
bar1
```

## Read past version of keys

Applications may want to read superseded versions of a key. For example, an application may wish to roll back to an old configuration by accessing an earlier version of a key. Alternatively, an application may want a consistent view over multiple keys through multiple requests by accessing key history.
Since every modification to the etcd cluster key-value store increments the global revision of an etcd cluster, an application can read superseded keys by providing an older etcd revision.

Suppose an etcd cluster already has the following keys:

```bash
foo = bar         # revision = 2
foo1 = bar1       # revision = 3
foo = bar_new     # revision = 4
foo1 = bar1_new   # revision = 5
```

Here are an example to access the past versions of keys:

```bash
$ etcdctl get --prefix foo # access the most recent versions of keys
foo
bar_new
foo1
bar1_new

$ etcdctl get --prefix --rev=4 foo # access the versions of keys at revision 4
foo
bar_new
foo1
bar1

$ etcdctl get --prefix --rev=3 foo # access the versions of keys at revision 3
foo
bar
foo1
bar1

$ etcdctl get --prefix --rev=2 foo # access the versions of keys at revision 2
foo
bar

$ etcdctl get --prefix --rev=1 foo # access the versions of keys at revision 1
```

## Read keys which are greater than or equal to the byte value of the specified key

Applications may want to read keys which are greater than or equal to the byte value of the specified key.

Suppose an etcd cluster already has the following keys:

```bash
a = 123
b = 456
z = 789
```

Here is the command to read keys which are greater than or equal to the byte value of key `b` :

```bash
$ etcdctl get --from-key b
b
456
z
789
```

## Delete keys

Applications can delete a key or a range of keys from an etcd cluster.

Suppose an etcd cluster already has the following keys:

```bash
foo = bar
foo1 = bar1
foo3 = bar3
zoo = val
zoo1 = val1
zoo2 = val2
a = 123
b = 456
z = 789
```

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

Here is the command to delete key `zoo` with the deleted key value pair returned:

```bash
$ etcdctl del --prev-kv zoo 
1   # one key is deleted
zoo # deleted key
val # the value of the deleted key
```

Here is the command to delete keys having prefix as `zoo`:

```bash
$ etcdctl del --prefix zoo 
2 # two keys are deleted
```

Here is the command to delete keys which are greater than or equal to the byte value of key `b` :

```bash
$ etcdctl del --from-key b
2 # two keys are deleted
```

## Watch key changes

Applications can watch on a key or a range of keys to monitor for any updates.

Here is the command to watch on key `foo`:

```bash
$ etcdctl watch foo
# in another terminal: etcdctl put foo bar
PUT
foo
bar
```

Here is the command to watch on key `foo` in hex format:

```bash
$ etcdctl watch foo --hex
# in another terminal: etcdctl put foo bar
PUT
\x66\x6f\x6f          # Key
\x62\x61\x72          # Value
```

Here is the command to watch on a range key from `foo` to `foo9`:

```bash
$ etcdctl watch foo foo9
# in another terminal: etcdctl put foo bar
PUT
foo
bar
# in another terminal: etcdctl put foo1 bar1
PUT
foo1
bar1
```

Here is the command to watch on keys having prefix `foo`:

```bash
$ etcdctl watch --prefix foo
# in another terminal: etcdctl put foo bar
PUT
foo
bar
# in another terminal: etcdctl put fooz1 barz1
PUT
fooz1
barz1
```

Here is the command to watch on multiple keys `foo` and `zoo`:

```bash
$ etcdctl watch -i 
$ watch foo
$ watch zoo
# in another terminal: etcdctl put foo bar
PUT
foo
bar
# in another terminal: etcdctl put zoo val
PUT
zoo
val
```

## Watch historical changes of keys

Applications may want to watch for historical changes of keys in etcd. For example, an application may wish to receive all the modifications of a key; if the application stays connected to etcd, then `watch` is good enough. However, if the application or etcd fails, a change may happen during the failure, and the application will not receive the update in real time. To guarantee the update is delivered, the application must be able to watch for historical changes to keys. To do this, an application can specify a historical revision on a watch, just like reading past version of keys.

Suppose we finished the following sequence of operations:

```bash
$ etcdctl put foo bar         # revision = 2
OK
$ etcdctl put foo1 bar1       # revision = 3
OK
$ etcdctl put foo bar_new     # revision = 4
OK
$ etcdctl put foo1 bar1_new   # revision = 5
OK
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
```

```bash
# watch for changes on key `foo` since revision 3
$ etcdctl watch --rev=3 foo
PUT
foo
bar_new
```

Here is an example to watch only from the last historical change:

```bash
# watch for changes on key `foo` and return last revision value along with modified value
$ etcdctl watch --prev-kv foo
# in another terminal: etcdctl put foo bar_latest
PUT
foo         # key
bar_new     # last value of foo key before modification
foo         # key
bar_latest  # value of foo key after modification
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

Note: The current revision of etcd server can be found using get command on any key (existent or non-existent) in json format. Example is shown below for mykey which does not exist in etcd server:

```bash
$ etcdctl get mykey -w=json
{"header":{"cluster_id":14841639068965178418,"member_id":10276657743932975437,"revision":15,"raft_term":4}}
```

## Grant leases

Applications can grant leases for keys from an etcd cluster. When a key is attached to a lease, its lifetime is bound to the lease's lifetime which in turn is governed by a time-to-live (TTL). Each lease has a minimum time-to-live (TTL) value specified by the application at grant time. The lease's actual TTL value is at least the minimum TTL and is chosen by the etcd cluster. Once a lease's TTL elapses, the lease expires and all attached keys are deleted.

Here is the command to grant a lease:

```bash
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

```bash
$ etcdctl lease grant 10
lease 32695410dcc0ca06 granted with TTL(10s)
$ etcdctl put --lease=32695410dcc0ca06 foo bar
OK
```

Here is the command to revoke the same lease:

```bash
$ etcdctl lease revoke 32695410dcc0ca06
lease 32695410dcc0ca06 revoked

$ etcdctl get foo
# empty response since foo is deleted due to lease revocation
```

## Keep leases alive

Applications can keep a lease alive by refreshing its TTL so it does not expire.

Suppose we finished the following sequence of operations:

```bash
$ etcdctl lease grant 10
lease 32695410dcc0ca06 granted with TTL(10s)
```

Here is the command to keep the same lease alive:

```bash
$ etcdctl lease keep-alive 32695410dcc0ca06
lease 32695410dcc0ca06 keepalived with TTL(100)
lease 32695410dcc0ca06 keepalived with TTL(100)
lease 32695410dcc0ca06 keepalived with TTL(100)
...
```

## Get lease information

Applications may want to know about lease information, so that they can be renewed or to check if the lease still exists or it has expired. Applications may also want to know the keys to which a particular lease is attached.

Suppose we finished the following sequence of operations:

```bash
# grant a lease with 500 second TTL
$ etcdctl lease grant 500
lease 694d5765fc71500b granted with TTL(500s)

# attach key zoo1 to lease 694d5765fc71500b
$ etcdctl put zoo1 val1 --lease=694d5765fc71500b
OK

# attach key zoo2 to lease 694d5765fc71500b
$ etcdctl put zoo2 val2 --lease=694d5765fc71500b
OK
```

Here is the command to get information about the lease:

```bash
$ etcdctl lease timetolive 694d5765fc71500b
lease 694d5765fc71500b granted with TTL(500s), remaining(258s)
```

Here is the command to get information about the lease along with the keys attached with the lease:

```bash
$ etcdctl lease timetolive --keys 694d5765fc71500b
lease 694d5765fc71500b granted with TTL(500s), remaining(132s), attached keys([zoo2 zoo1])

# if the lease has expired or does not exist it will give the below response:
Error:  etcdserver: requested lease not found
```

