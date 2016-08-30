# go-memdb

Provides the `memdb` package that implements a simple in-memory database
built on immutable radix trees. The database provides Atomicity, Consistency
and Isolation from ACID. Being that it is in-memory, it does not provide durability.
The database is instantiated with a schema that specifies the tables and indicies
that exist and allows transactions to be executed.

The database provides the following:

* Multi-Version Concurrency Control (MVCC) - By leveraging immutable radix trees
  the database is able to support any number of concurrent readers without locking,
  and allows a writer to make progress.

* Transaction Support - The database allows for rich transactions, in which multiple
  objects are inserted, updated or deleted. The transactions can span multiple tables,
  and are applied atomically. The database provides atomicity and isolation in ACID
  terminology, such that until commit the updates are not visible.

* Rich Indexing - Tables can support any number of indexes, which can be simple like
  a single field index, or more advanced compound field indexes. Certain types like
  UUID can be efficiently compressed from strings into byte indexes for reduces
  storage requirements.

For the underlying immutable radix trees, see [go-immutable-radix](https://github.com/hashicorp/go-immutable-radix).

Documentation
=============

The full documentation is available on [Godoc](http://godoc.org/github.com/hashicorp/go-memdb).

Example
=======

Below is a simple example of usage

```go
// Create a sample struct
type Person struct {
    Email string
    Name  string
    Age   int
}

// Create the DB schema
schema := &memdb.DBSchema{
    Tables: map[string]*memdb.TableSchema{
        "person": &memdb.TableSchema{
            Name: "person",
            Indexes: map[string]*memdb.IndexSchema{
                "id": &memdb.IndexSchema{
                    Name:    "id",
                    Unique:  true,
                    Indexer: &memdb.StringFieldIndex{Field: "Email"},
                },
            },
        },
    },
}

// Create a new data base
db, err := memdb.NewMemDB(schema)
if err != nil {
    panic(err)
}

// Create a write transaction
txn := db.Txn(true)

// Insert a new person
p := &Person{"joe@aol.com", "Joe", 30}
if err := txn.Insert("person", p); err != nil {
    panic(err)
}

// Commit the transaction
txn.Commit()

// Create read-only transaction
txn = db.Txn(false)
defer txn.Abort()

// Lookup by email
raw, err := txn.First("person", "id", "joe@aol.com")
if err != nil {
    panic(err)
}

// Say hi!
fmt.Printf("Hello %s!", raw.(*Person).Name)

```

