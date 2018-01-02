# etcd3 API

TODO: API doc

## Data Model

etcd is designed to reliably store infrequently updated data and provide reliable watch queries. etcd exposes previous versions of key-value pairs to support inexpensive snapshots and watch history events (“time travel queries”). A persistent, multi-version, concurrency-control data model is a good fit for these use cases.

etcd stores data in a multiversion [persistent][persistent-ds] key-value store. The persistent key-value store preserves the previous version of a key-value pair when its value is superseded with new data. The key-value store is effectively immutable; its operations do not update the structure in-place, but instead always generates a new updated structure. All past versions of keys are still accessible and watchable after modification. To prevent the data store from growing indefinitely over time from maintaining old versions, the store may be compacted to shed the oldest versions of superseded data.

### Logical View

The store’s logical view is a flat binary key space. The key space has a lexically sorted index on byte string keys so range queries are inexpensive.

The key space maintains multiple revisions. Each atomic mutative operation (e.g., a transaction operation may contain multiple operations) creates a new revision on the key space. All data held by previous revisions remains unchanged. Old versions of key can still be accessed through previous revisions. Likewise, revisions are indexed as well; ranging over revisions with watchers is efficient. If the store is compacted to recover space, revisions before the compact revision will be removed.

A key’s lifetime spans a generation. Each key may have one or multiple generations. Creating a key increments the generation of that key, starting at 1 if the key never existed. Deleting a key generates a key tombstone, concluding the key’s current generation. Each modification of a key creates a new version of the key. Once a compaction happens, any generation ended before the given revision will be removed and values set before the compaction revision except the latest one will be removed.

### Physical View

etcd stores the physical data as key-value pairs in a persistent [b+tree][b+tree]. Each revision of the store’s state only contains the delta from its previous revision to be efficient. A single revision may correspond to multiple keys in the tree.

The key of key-value pair is a 3-tuple (major, sub, type). Major is the store revision holding the key. Sub differentiates among  keys within the same revision. Type is an optional suffix for special value (e.g., `t` if the value contains a tombstone). The value of the key-value pair contains the modification from previous revision, thus one delta from previous revision. The b+tree is ordered by key in lexical byte-order. Ranged lookups over revision deltas are fast; this enables quickly finding modifications from one specific revision to another. Compaction removes out-of-date keys-value pairs.

etcd also keeps a secondary in-memory [btree][btree] index to speed up range queries over keys. The keys in the btree index are the keys of the store exposed to user. The value is a pointer to the modification of the persistent b+tree. Compaction removes dead pointers.

## KV API Guarantees

etcd is a consistent and durable key value store with mini-transaction(TODO: link to txn doc when we have it) support. The key value store is exposed through the KV APIs. etcd tries to ensure the strongest consistency and durability guarantees for a distributed system. This specification enumerates the KV API guarantees made by etcd.

### APIs to consider

* Read APIs
    * range
    * watch
* Write APIs
    * put
    * delete
* Combination (read-modify-write) APIs
    * txn

### etcd Specific Definitions

#### operation completed

An etcd operation is considered complete when it is committed through consensus, and therefore “executed” -- permanently stored -- by the etcd storage engine. The client knows an operation is completed when it receives a response from the etcd server. Note that the client may be uncertain about the status of an operation if it times out, or there is a network disruption between the client and the etcd member. etcd may also abort operations when there is a leader election. etcd does not send `abort` responses to  clients’ outstanding requests in this event.

#### revision

An etcd operation that modifies the key value store is assigned with a single increasing revision. A transaction operation might modifies the key value store multiple times, but only one revision is assigned. The revision attribute of a key value pair that modified by the operation has the same value as the revision of the operation. The revision can be used as a logical clock for key value store. A key value pair that has a larger revision is modified after a key value pair with a smaller revision. Two key value pairs that have the same revision are modified by an operation "concurrently".

### Guarantees Provided

#### Atomicity

All API requests are atomic; an operation either completes entirely or not at all. For watch requests, all events generated by one operation will be in one watch response. Watch never observes partial events for a single operation.

#### Consistency

All API calls ensure [sequential consistency][seq_consistency], the strongest consistency guarantee available from distributed systems. No matter which etcd member server a client makes requests to, a client reads the same events in the same order. If two members complete the same number of operations, the state of the two members is consistent.

For watch operations, etcd guarantees to return the same value for the same key across all members for the same revision. For range operations, etcd has a similar guarantee for [linearized][Linearizability] access; serialized access may be behind the quorum state, so that the later revision is not yet available.

As with all distributed systems, it is impossible for etcd to ensure [strict consistency][strict_consistency]. etcd does not guarantee that it will return to a read the “most recent” value (as measured by a wall clock when a request is completed) available on any cluster member.

#### Isolation

etcd ensures [serializable isolation][serializable_isolation], which is the highest isolation level available in distributed systems. Read operations will never observe any intermediate data.

#### Durability

Any completed operations are durable. All accessible data is also durable data. A read will never return data that has not been made durable.

#### Linearizability

Linearizability (also known as Atomic Consistency or External Consistency) is a consistency level between strict consistency and sequential consistency.

For linearizability, suppose each operation receives a timestamp from a loosely synchronized global clock. Operations are linearized if and only if they always complete as though they were executed in a sequential order and each operation appears to complete in the order specified by the program. Likewise, if an operation’s timestamp precedes another, that operation must also precede the other operation in the sequence.

For example, consider a client completing a write at time point 1 (*t1*). A client issuing a read at *t2* (for *t2* > *t1*) should receive a value at least as recent as the previous write, completed at *t1*. However, the read might actually complete only by *t3*, and the returned value, current at *t2* when the read began, might be "stale" by *t3*.

etcd does not ensure linearizability for watch operations. Users are expected to verify the revision of watch responses to ensure correct ordering.

etcd ensures linearizability for all other operations by default. Linearizability comes with a cost, however, because linearized requests must go through the Raft consensus process. To obtain lower latencies and higher throughput for read requests, clients can configure a request’s consistency mode to `serializable`, which may access stale data with respect to quorum, but removes the performance penalty of linearized accesses' reliance on live consensus.

[persistent-ds]: https://en.wikipedia.org/wiki/Persistent_data_structure
[btree]: https://en.wikipedia.org/wiki/B-tree
[b+tree]: https://en.wikipedia.org/wiki/B%2B_tree
[seq_consistency]: https://en.wikipedia.org/wiki/Consistency_model#Sequential_consistency
[strict_consistency]: https://en.wikipedia.org/wiki/Consistency_model#Strict_consistency
[serializable_isolation]: https://en.wikipedia.org/wiki/Isolation_(database_systems)#Serializable
[Linearizability]: #linearizability
