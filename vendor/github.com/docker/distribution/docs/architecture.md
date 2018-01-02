---
published: false
---

# Architecture

## Design
**TODO(stevvooe):** Discuss the architecture of the registry, internally and externally, in a few different deployment scenarios.

### Eventual Consistency

> **NOTE:** This section belongs somewhere, perhaps in a design document. We
> are leaving this here so the information is not lost.

Running the registry on eventually consistent backends has been part of the
design from the beginning. This section covers some of the approaches to
dealing with this reality.

There are a few classes of issues that we need to worry about when
implementing something on top of the storage drivers:

1. Read-After-Write consistency (see this [article on
   s3](http://shlomoswidler.com/2009/12/read-after-write-consistency-in-amazon.html)).
2. [Write-Write Conflicts](http://en.wikipedia.org/wiki/Write%E2%80%93write_conflict).

In reality, the registry must worry about these kinds of errors when doing the
following:

1. Accepting data into a temporary upload file may not have latest data block
   yet (read-after-write).
2. Moving uploaded data into its blob location (write-write race).
3. Modifying the "current" manifest for given tag (write-write race).
4. A whole slew of operations around deletes (read-after-write, delete-write
   races, garbage collection, etc.).

The backend path layout employs a few techniques to avoid these problems:

1. Large writes are done to private upload directories. This alleviates most
   of the corruption potential under multiple writers by avoiding multiple
   writers.
2. Constraints in storage driver implementations, such as support for writing
   after the end of a file to extend it.
3. Digest verification to avoid data corruption.
4. Manifest files are stored by digest and cannot change.
5. All other non-content files (links, hashes, etc.) are written as an atomic
   unit. Anything that requires additions and deletions is broken out into
   separate "files". Last writer still wins.

Unfortunately, one must play this game when trying to build something like
this on top of eventually consistent storage systems. If we run into serious
problems, we can wrap the storagedrivers in a shared consistency layer but
that would increase complexity and hinder registry cluster performance.
