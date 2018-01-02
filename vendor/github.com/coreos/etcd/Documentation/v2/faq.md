# FAQ

## 1) Why can an etcd client read an old version of data when a majority of the etcd cluster members are down?

In situations where a client connects to a minority, etcd
favors by default availability over consistency. This means that even though
data might be “out of date”, it is still better to return something versus
nothing.

In order to confirm that a read is up to date with a majority of the cluster,
the client can use the `quorum=true` parameter on reads of keys. This means
that a majority of the cluster is checked on reads before returning the data,
otherwise the read will timeout and fail.

## 2) With quorum=false, doesn’t this mean that if my client switched the member it was connected to, that it could experience a logical ordering where the cluster goes backwards in time?

Yes, but this could be handled at the etcd client implementation via
remembering the last seen index. The “index” is the cluster's single
irrevocable sequence of the entire modification history. The client could
remember the last seen index, and determine via comparing the index returned on
the GET whether or not the state of the key-value pair is before or after its
last seen state.

## 3) What happens if a watch is registered on a minority member?

The watch will stay untriggered, even as modifications are occurring in the
majority quorum. This is an open issue, and is being addressed in v3. There are
multiple ways to work around the watch trigger not firing.

1) build a signaling mechanism independent of etcd. This could be as simple as
a “pulse” to the client to reissue a GET with quorum=true for the most recent
version of the data.

2) poll on the `/v2/keys` endpoint and check that the raft-index is increasing every
timeout.

## 4) What is a proxy used for?

A proxy is a redirection server to the etcd cluster. The proxy handles the
redirection of a client to the current configuration of the etcd cluster. A
typical use case is to start a proxy on a machine, and on first boot up of the
proxy specify both the `--proxy` flag and the `--initial-cluster` flag.

From there, any etcdctl client that starts up automatically speaks to the local
proxy and the proxy redirects operations to the current configuration of the
cluster it was originally paired with.

In the v2 spec of etcd, proxies cannot be promoted to members of the cluster.
They also cannot be promoted to followers or at any point become part of the
replication of the etcd cluster itself.

## 5) How is cluster membership and health handled in etcd v2?

The design goal of etcd is that reconfiguration is simply an API, and health
monitoring and addition/removal of members is up to the individual application
and their integration with the reconfiguration API.

Thus, a member that is down, even infinitely, will never be automatically
removed from the etcd cluster member list.

This makes sense because it's usually an application level / administrative
action to determine whether a reconfiguration should happen based on health.

For more information, refer to the [runtime reconfiguration design document][runtime-reconf-design].

## 6) how does --endpoint work with etcdctl?

The `--endpoint` flag can specify any number of etcd cluster members in a comma
separated list. This list might be a subset, equal to, or more than the actual
etcd cluster member list itself.

If only one peer is specified via the `--endpoint` flag, the etcdctl discovers the
rest of the cluster via the member list of that one peer, and then it randomly
chooses a member to use.  Again, the client can use the `quorum=true` flag on
reads, which will always fail when using a member in the minority.

If peers from multiple clusters are specified via the `--endpoint` flag, etcdctl
will randomly choose a peer, and the request will simply get routed to one of
the clusters. This is probably not what you want.

Note: --peers flag is now deprecated and --endpoint should be used instead,
as it might confuse users to give etcdctl a peerURL.

[runtime-reconf-design]: runtime-reconf-design.md
