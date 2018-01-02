# Glossary

This document defines the various terms used in etcd documentation, command line and source code.

## Alarm

The etcd server raises an alarm whenever the cluster needs operator intervention to remain reliable.

## Authentication

Authentication manages user access permissions for etcd resources.

## Client

A client connects to the etcd cluster to issue service requests such as fetching key-value pairs, writing data, or watching for updates.

## Cluster

Cluster consists of several members.

The node in each member follows raft consensus protocol to replicate logs. Cluster receives proposals from members, commits them and apply to local store.

## Compaction

Compaction discards all etcd event history and superseded keys prior to a given revision. It is used to reclaim storage space in the etcd backend database.

## Election

The etcd cluster holds elections among its members to choose a leader as part of the raft consensus protocol.

## Endpoint

A URL pointing to an etcd service or resource.

## Key

A user-defined identifier for storing and retrieving user-defined values in etcd.

## Key range

A set of keys containing either an individual key, a lexical interval for all x such that a < x <= b, or all keys greater than a given key.

## Keyspace

The set of all keys in an etcd cluster.

## Lease

A short-lived renewable contract that deletes keys associated with it on its expiry.

## Member

A logical etcd server that participates in serving an etcd cluster.

## Modification Revision

The first revision to hold the last write to a given key.

## Peer

Peer is another member of the same cluster.

## Proposal

A proposal is a request (for example a write request, a configuration change request) that needs to go through raft protocol.

## Quorum

The number of active members needed for consensus to modify the cluster state. etcd requires a member majority to reach quorum.

## Revision

A 64-bit cluster-wide counter that is incremented each time the keyspace is modified.

## Role

A unit of permissions over a set of key ranges which may be granted to a set of users for access control.

## Snapshot

A point-in-time backup of the etcd cluster state.

## Store

The physical storage backing the cluster keyspace.

## Transaction

An atomically executed set of operations. All modified keys in a transaction share the same modification revision.

## Key Version

The number of writes to a key since it was created, starting at 1. The version of a nonexistent or deleted key is 0.

## Watcher

A client opens a watcher to observe updates on a given key range.
