# etcd3 API

NOTE: this doc is not finished!

## Response header

All Responses from etcd API have a [response header][response_header] attached. The response header includes the metadata of the response.

```proto
message ResponseHeader {
  uint64 cluster_id = 1;
  uint64 member_id = 2;
  int64 revision = 3;
  uint64 raft_term = 4;
}
```

* Cluster_ID - the ID of the cluster that generates the response
* Member_ID - the ID of the member that generates the response
* Revision - the revision of the key-value store when the response is generated
* Raft_Term - the Raft term of the member when the response is generated

An application may read the Cluster_ID (Member_ID) field to ensure it is communicating with the intended cluster (member).

Applications can use the `Revision` to know the latest revision of the key-value store. This is especially useful when applications specify a historical revision to make time `travel query` and wishes to know the latest revision at the time of the request.

Applications can use `Raft_Term` to detect when the cluster completes a new leader election.

## Key-Value API

Key-Value API is used to manipulate key-value pairs stored inside etcd. The key-value API is defined as a [gRPC service][kv-service]. The Key-Value pair is defined as structured data in [protobuf format][kv-proto].

### Key-Value pair

A key-value pair is the smallest unit that the key-value API can manipulate. Each key-value pair has a number of fields:

```protobuf
message KeyValue {
  bytes key = 1;
  int64 create_revision = 2;
  int64 mod_revision = 3;
  int64 version = 4;
  bytes value = 5;
  int64 lease = 6;
}
```

* Key - key in bytes. An empty key is not allowed.
* Value - value in bytes.
* Version - version is the version of the key. A deletion resets the version to zero and any modification of the key increases its version.
* Create_Revision - revision of the last creation on the key.
* Mod_Revision - revision of the last modification on the key.
* Lease - the ID of the lease attached to the key. If lease is 0, then no lease is attached to the key.

[kv-proto]: https://github.com/coreos/etcd/blob/master/mvcc/mvccpb/kv.proto
[kv-service]: https://github.com/coreos/etcd/blob/master/etcdserver/etcdserverpb/rpc.proto
[response_header]: https://github.com/coreos/etcd/blob/master/etcdserver/etcdserverpb/rpc.proto
