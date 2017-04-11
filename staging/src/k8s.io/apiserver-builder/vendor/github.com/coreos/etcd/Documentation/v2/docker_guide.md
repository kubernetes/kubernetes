# Running etcd under Docker

The following guide will show you how to run etcd under Docker using the [static bootstrap process](clustering.md#static).

## Running etcd in standalone mode

In order to expose the etcd API to clients outside of the Docker host you'll need use the host IP address when configuring etcd.

```
export HostIP="192.168.12.50"
```

The following `docker run` command will expose the etcd client API over ports 4001 and 2379, and expose the peer port over 2380.

This will run the latest release version of etcd. You can specify version if needed (e.g. `quay.io/coreos/etcd:v2.2.0`).

```
docker run -d -v /usr/share/ca-certificates/:/etc/ssl/certs -p 4001:4001 -p 2380:2380 -p 2379:2379 \
 --name etcd quay.io/coreos/etcd \
 -name etcd0 \
 -advertise-client-urls http://${HostIP}:2379,http://${HostIP}:4001 \
 -listen-client-urls http://0.0.0.0:2379,http://0.0.0.0:4001 \
 -initial-advertise-peer-urls http://${HostIP}:2380 \
 -listen-peer-urls http://0.0.0.0:2380 \
 -initial-cluster-token etcd-cluster-1 \
 -initial-cluster etcd0=http://${HostIP}:2380 \
 -initial-cluster-state new
```

Configure etcd clients to use the Docker host IP and one of the listening ports from above.

```
etcdctl -C http://192.168.12.50:2379 member list
```

```
etcdctl -C http://192.168.12.50:4001 member list
```

## Running a 3 node etcd cluster

Using Docker to setup a multi-node cluster is very similar to the standalone mode configuration.
The main difference being the value used for the `-initial-cluster` flag, which must contain the peer urls for each etcd member in the cluster.

### etcd0

```
docker run -d -v /usr/share/ca-certificates/:/etc/ssl/certs -p 4001:4001 -p 2380:2380 -p 2379:2379 \
 --name etcd quay.io/coreos/etcd \
 -name etcd0 \
 -advertise-client-urls http://192.168.12.50:2379,http://192.168.12.50:4001 \
 -listen-client-urls http://0.0.0.0:2379,http://0.0.0.0:4001 \
 -initial-advertise-peer-urls http://192.168.12.50:2380 \
 -listen-peer-urls http://0.0.0.0:2380 \
 -initial-cluster-token etcd-cluster-1 \
 -initial-cluster etcd0=http://192.168.12.50:2380,etcd1=http://192.168.12.51:2380,etcd2=http://192.168.12.52:2380 \
 -initial-cluster-state new
```

### etcd1

```
docker run -d -v /usr/share/ca-certificates/:/etc/ssl/certs -p 4001:4001 -p 2380:2380 -p 2379:2379 \
 --name etcd quay.io/coreos/etcd \
 -name etcd1 \
 -advertise-client-urls http://192.168.12.51:2379,http://192.168.12.51:4001 \
 -listen-client-urls http://0.0.0.0:2379,http://0.0.0.0:4001 \
 -initial-advertise-peer-urls http://192.168.12.51:2380 \
 -listen-peer-urls http://0.0.0.0:2380 \
 -initial-cluster-token etcd-cluster-1 \
 -initial-cluster etcd0=http://192.168.12.50:2380,etcd1=http://192.168.12.51:2380,etcd2=http://192.168.12.52:2380 \
 -initial-cluster-state new
```

### etcd2

```
docker run -d -v /usr/share/ca-certificates/:/etc/ssl/certs -p 4001:4001 -p 2380:2380 -p 2379:2379 \
 --name etcd quay.io/coreos/etcd \
 -name etcd2 \
 -advertise-client-urls http://192.168.12.52:2379,http://192.168.12.52:4001 \
 -listen-client-urls http://0.0.0.0:2379,http://0.0.0.0:4001 \
 -initial-advertise-peer-urls http://192.168.12.52:2380 \
 -listen-peer-urls http://0.0.0.0:2380 \
 -initial-cluster-token etcd-cluster-1 \
 -initial-cluster etcd0=http://192.168.12.50:2380,etcd1=http://192.168.12.51:2380,etcd2=http://192.168.12.52:2380 \
 -initial-cluster-state new
```

Once the cluster has been bootstrapped etcd clients can be configured with a list of etcd members:

```
etcdctl -C http://192.168.12.50:2379,http://192.168.12.51:2379,http://192.168.12.52:2379 member list
```
