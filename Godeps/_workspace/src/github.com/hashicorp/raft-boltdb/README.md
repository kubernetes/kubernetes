raft-boltdb
===========

This repository provides the `raftboltdb` package. The package exports the
`BoltStore` which is an implementation of both a `LogStore` and `StableStore`.

It is meant to be used as a backend for the `raft` [package
here](https://github.com/hashicorp/raft).

This implementation uses [BoltDB](https://github.com/boltdb/bolt). BoltDB is
a simple key/value store implemented in pure Go, and inspired by LMDB.
