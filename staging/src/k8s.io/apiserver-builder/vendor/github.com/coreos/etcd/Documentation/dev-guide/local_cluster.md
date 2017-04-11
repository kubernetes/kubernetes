# Setup a local cluster

For testing and development deployments, the quickest and easiest way is to set up a local cluster. For a production deployment, refer to the [clustering][clustering] section.

## Local standalone cluster

Deploying an etcd cluster as a standalone cluster is straightforward. Start it with just one command:

```
$ ./etcd
...
```

The started etcd member listens on `localhost:2379` for client requests.

To interact with the started cluster by using etcdctl:

```
# use API version 3
$ export ETCDCTL_API=3

$ ./etcdctl put foo bar
OK

$ ./etcdctl get foo
bar
```

## Local multi-member cluster

A Procfile is provided to easily set up a local multi-member cluster. Start a multi-member cluster with a few commands:

```
# install goreman program to control Profile-based applications.
$ go get github.com/mattn/goreman
$ goreman -f Procfile start
...
```

The started members listen on `localhost:12379`, `localhost:22379`, and `localhost:32379` for client requests respectively.

To interact with the started cluster by using etcdctl:

```
# use API version 3
$ export ETCDCTL_API=3

$ etcdctl --write-out=table --endpoints=localhost:12379 member list
+------------------+---------+--------+------------------------+------------------------+
|        ID        | STATUS  |  NAME  |       PEER ADDRS       |      CLIENT ADDRS      |
+------------------+---------+--------+------------------------+------------------------+
| 8211f1d0f64f3269 | started | infra1 | http://127.0.0.1:12380 | http://127.0.0.1:12379 |
| 91bc3c398fb3c146 | started | infra2 | http://127.0.0.1:22380 | http://127.0.0.1:22379 |
| fd422379fda50e48 | started | infra3 | http://127.0.0.1:32380 | http://127.0.0.1:32379 |
+------------------+---------+--------+------------------------+------------------------+

$ etcdctl --endpoints=localhost:12379 put foo bar
OK
```

To exercise etcd's fault tolerance, kill a member:

```
# kill etcd2
$ goreman run stop etcd2

$ etcdctl --endpoints=localhost:12379 put key hello
OK

$ etcdctl --endpoints=localhost:12379 get key
hello

# try to get key from the killed member
$ etcdctl --endpoints=localhost:22379 get key
2016/04/18 23:07:35 grpc: Conn.resetTransport failed to create client transport: connection error: desc = "transport: dial tcp 127.0.0.1:22379: getsockopt: connection refused"; Reconnecting to "localhost:22379"
Error:  grpc: timed out trying to connect

# restart the killed member
$ goreman run restart etcd2

# get the key from restarted member
$ etcdctl --endpoints=localhost:22379 get key
hello
```

To learn more about interacting with etcd, read [interacting with etcd section][interacting].

[interacting]: ./interacting_v3.md
[clustering]: ../op-guide/clustering.md

