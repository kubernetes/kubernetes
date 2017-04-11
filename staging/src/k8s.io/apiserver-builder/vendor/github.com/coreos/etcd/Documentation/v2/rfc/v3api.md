# Overview

The etcd v3 API is designed to give users a more efficient and cleaner abstraction compared to etcd v2. There are a number of semantic and protocol changes in this new API. For an overview [see Xiang Li's video](https://youtu.be/J5AioGtEPeQ?t=211).

To prove out the design of the v3 API the team has also built [a number of example recipes](https://github.com/coreos/etcd/tree/master/contrib/recipes), there is a [video discussing these recipes too](https://www.youtube.com/watch?v=fj-2RY-3yVU&feature=youtu.be&t=590).

# Design

1. Flatten binary key-value space
    
2. Keep the event history until compaction
    - access to old version of keys
    - user controlled history compaction
    
3. Support range query
    - Pagination support with limit argument
    - Support consistency guarantee across multiple range queries
    
4. Replace TTL key with Lease
    - more efficient/ low cost keep alive
    - a logical group of TTL keys
    
5. Replace CAS/CAD with multi-object Txn
    - MUCH MORE powerful and flexible
    
6. Support efficient watching with multiple ranges

7. RPC API supports the completed set of APIs. 
    - more efficient than JSON/HTTP
    - additional txn/lease support

8. HTTP API supports a subset of APIs.
    - easy for people to try out etcd
    - easy for people to write simple etcd application


## Notes

### Request Size Limitation

The max request size is around 1MB. Since etcd replicates requests in a streaming fashion, a very large
request might block other requests for a long time. The use case for etcd is to store small configuration
values, so we prevent user from submitting large requests. This also applies to Txn requests. We might loosen
the size in the future a little bit or make it configurable.

## Protobuf Defined API

[api protobuf][api-protobuf]

[kv protobuf][kv-protobuf]

## Examples

### Put a key (foo=bar)
```
// A put is always successful
Put( PutRequest { key = foo, value = bar } )

PutResponse { 
    cluster_id = 0x1000,
    member_id = 0x1,
    revision = 1,
    raft_term = 0x1,
}
```

### Get a key (assume we have foo=bar)
```
Get ( RangeRequest { key = foo } )

RangeResponse {
    cluster_id = 0x1000,
    member_id = 0x1,
    revision = 1,
    raft_term = 0x1,
    kvs = {
      {
          key = foo,
          value = bar,
          create_revision = 1,
          mod_revision = 1,
          version = 1;
      },
    },
}
```

### Range over a key space (assume we have foo0=bar0… foo100=bar100)
```
Range ( RangeRequest { key = foo, end_key = foo80, limit = 30  } )

RangeResponse {
    cluster_id = 0x1000,
    member_id = 0x1,
    revision = 100,
    raft_term = 0x1,
    kvs = {
      {
          key = foo0,
          value = bar0,
          create_revision = 1,
          mod_revision = 1,
          version = 1;
      },
         ...,
      {
          key = foo30,
          value = bar30,
          create_revision = 30,
          mod_revision = 30,
          version = 1;
      },
    },
}
```

### Finish a txn (assume we have foo0=bar0, foo1=bar1)
```
Txn(TxnRequest {
    // mod_revision of foo0 is equal to 1, mod_revision of foo1 is greater than 1
    compare = {
        {compareType = equal, key = foo0, mod_revision = 1}, 
        {compareType = greater, key = foo1, mod_revision = 1}}
    },
    // if the comparison succeeds, put foo2 = bar2
    success = {PutRequest { key = foo2, value = success }},
    // if the comparison fails, put foo2=fail
    failure = {PutRequest { key = foo2, value = failure }},
)

TxnResponse {
    cluster_id = 0x1000,
    member_id = 0x1,
    revision = 3,
    raft_term = 0x1,
    succeeded = true,
    responses = {
      // response of PUT foo2=success
      {
            cluster_id = 0x1000,
            member_id = 0x1,
            revision = 3,
            raft_term = 0x1,
        }
    }
}
```

### Watch on a key/range

```
Watch( WatchRequest{
           key = foo,
           end_key = fop, // prefix foo
           start_revision = 20,
           end_revision = 10000,
           // server decided notification frequency
           progress_notification = true,
       } 
       … // this can be a watch request stream
      )

// put (foo0=bar0) event at 3
WatchResponse {
    cluster_id = 0x1000,
    member_id = 0x1,
    revision = 3,
    raft_term = 0x1,
    event_type = put,
    kv = {
              key = foo0,
              value = bar0,
              create_revision = 1,
              mod_revision = 1,
              version = 1;
          },
    }
    …
    
    // a notification at 2000
    WatchResponse {
        cluster_id = 0x1000,
        member_id = 0x1,
        revision = 2000,
        raft_term = 0x1,
        // nil event as notification
    }
    
    … 
    
    // put (foo0=bar3000) event at 3000
    WatchResponse {
        cluster_id = 0x1000,
        member_id = 0x1,
        revision = 3000,
        raft_term = 0x1,
        event_type = put,
        kv = {
                key = foo0,
                value = bar3000,
                create_revision = 1,
                mod_revision = 3000,
                version = 2;
          },
    }
    …
    
```

[api-protobuf]: https://github.com/coreos/etcd/blob/master/etcdserver/etcdserverpb/rpc.proto
[kv-protobuf]: https://github.com/coreos/etcd/blob/master/storage/storagepb/kv.proto
