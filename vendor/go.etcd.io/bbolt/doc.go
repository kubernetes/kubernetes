/*
package bbolt implements a low-level key/value store in pure Go. It supports
fully serializable transactions, ACID semantics, and lock-free MVCC with
multiple readers and a single writer. Bolt can be used for projects that
want a simple data store without the need to add large dependencies such as
Postgres or MySQL.

Bolt is a single-level, zero-copy, B+tree data store. This means that Bolt is
optimized for fast read access and does not require recovery in the event of a
system crash. Transactions which have not finished committing will simply be
rolled back in the event of a crash.

The design of Bolt is based on Howard Chu's LMDB database project.

Bolt currently works on Windows, Mac OS X, and Linux.


Basics

There are only a few types in Bolt: DB, Bucket, Tx, and Cursor. The DB is
a collection of buckets and is represented by a single file on disk. A bucket is
a collection of unique keys that are associated with values.

Transactions provide either read-only or read-write access to the database.
Read-only transactions can retrieve key/value pairs and can use Cursors to
iterate over the dataset sequentially. Read-write transactions can create and
delete buckets and can insert and remove keys. Only one read-write transaction
is allowed at a time.


Caveats

The database uses a read-only, memory-mapped data file to ensure that
applications cannot corrupt the database, however, this means that keys and
values returned from Bolt cannot be changed. Writing to a read-only byte slice
will cause Go to panic.

Keys and values retrieved from the database are only valid for the life of
the transaction. When used outside the transaction, these byte slices can
point to different data or can point to invalid memory which will cause a panic.


*/
package bbolt
