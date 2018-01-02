# Backward Compatibility

The main goal of etcd 2.0 release is to improve cluster safety around bootstrapping and dynamic reconfiguration. To do this, we deprecated the old error-prone APIs and provide a new set of APIs.

The other main focus of this release was a more reliable Raft implementation, but as this change is internal it should not have any notable effects to users.

## Command Line Flags Changes

The major flag changes are to mostly related to bootstrapping. The `initial-*` flags provide an improved way to specify the required criteria to start the cluster. The advertised URLs now support a list of values instead of a single value, which allows etcd users to gracefully migrate to the new set of IANA-assigned ports (2379/client and 2380/peers) while maintaining backward compatibility with the old ports.

 - `-addr` is replaced by `-advertise-client-urls`.
 - `-bind-addr` is replaced by `-listen-client-urls`.
 - `-peer-addr` is replaced by `-initial-advertise-peer-urls`.
 - `-peer-bind-addr` is replaced by `-listen-peer-urls`.
 - `-peers` is replaced by `-initial-cluster`.
 - `-peers-file` is replaced by `-initial-cluster`.
 - `-peer-heartbeat-interval` is replaced by `-heartbeat-interval`.
 - `-peer-election-timeout` is replaced by `-election-timeout`.

The documentation of new command line flags can be found at
https://github.com/coreos/etcd/blob/master/Documentation/v2/configuration.md.

## Data Directory Naming

The default data dir location has changed from {$hostname}.etcd to {name}.etcd.

## Key-Value API

### Read consistency flag

The consistent flag for read operations is removed in etcd 2.0.0. The normal read operations provides the same consistency guarantees with the 0.4.6 read operations with consistent flag set.

The read consistency guarantees are:

The consistent read guarantees the sequential consistency within one client that talks to one etcd server. Read/Write from one client to one etcd member should be observed in order. If one client write a value to an etcd server successfully, it should be able to get the value out of the server immediately.

Each etcd member will proxy the request to leader and only return the result to user after the result is applied on the local member. Thus after the write succeed, the user is guaranteed to see the value on the member it sent the request to.

Reads do not provide linearizability. If you want linearizable read, you need to set quorum option to true.

**Previous behavior**

We added an option for a consistent read in the old version of etcd since etcd 0.x redirects the write request to the leader. When the user get back the result from the leader, the member it sent the request to originally might not apply the write request yet. With the consistent flag set to true, the client will always send read request to the leader. So one client should be able to see its last write when consistent=true is enabled. There is no order guarantees among different clients.


## Standby

etcd 0.4’s standby mode has been deprecated. [Proxy mode][proxymode] is introduced to solve a subset of problems standby was solving.

Standby mode was intended for large clusters that had a subset of the members acting in the consensus process. Overall this process was too magical and allowed for operators to back themselves into a corner.

Proxy mode in 2.0 will provide similar functionality, and with improved control over which machines act as proxies due to the operator specifically configuring them. Proxies also support read only or read/write modes for increased security and durability.

[proxymode]: proxy.md

## Discovery Service

A size key needs to be provided inside a [discovery token][discoverytoken].

[discoverytoken]: clustering.md#custom-etcd-discovery-service

## HTTP Admin API

`v2/admin` on peer url and `v2/keys/_etcd` are unified under the new [v2/members API][members-api] to better explain which machines are part of an etcd cluster, and to simplify the keyspace for all your use cases.

[members-api]: members_api.md

## HTTP Key Value API
- The follower can now transparently proxy write requests to the leader. Clients will no longer see 307 redirections to the leader from etcd.

- Expiration time is in UTC instead of local time.

