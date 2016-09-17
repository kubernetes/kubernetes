package memdb

import (
	"bytes"
	"fmt"
	"strings"
	"sync/atomic"
	"unsafe"

	"github.com/hashicorp/go-immutable-radix"
)

const (
	id = "id"
)

// tableIndex is a tuple of (Table, Index) used for lookups
type tableIndex struct {
	Table string
	Index string
}

// Txn is a transaction against a MemDB.
// This can be a read or write transaction.
type Txn struct {
	db      *MemDB
	write   bool
	rootTxn *iradix.Txn
	after   []func()

	modified map[tableIndex]*iradix.Txn
}

// readableIndex returns a transaction usable for reading the given
// index in a table. If a write transaction is in progress, we may need
// to use an existing modified txn.
func (txn *Txn) readableIndex(table, index string) *iradix.Txn {
	// Look for existing transaction
	if txn.write && txn.modified != nil {
		key := tableIndex{table, index}
		exist, ok := txn.modified[key]
		if ok {
			return exist
		}
	}

	// Create a read transaction
	path := indexPath(table, index)
	raw, _ := txn.rootTxn.Get(path)
	indexTxn := raw.(*iradix.Tree).Txn()
	return indexTxn
}

// writableIndex returns a transaction usable for modifying the
// given index in a table.
func (txn *Txn) writableIndex(table, index string) *iradix.Txn {
	if txn.modified == nil {
		txn.modified = make(map[tableIndex]*iradix.Txn)
	}

	// Look for existing transaction
	key := tableIndex{table, index}
	exist, ok := txn.modified[key]
	if ok {
		return exist
	}

	// Start a new transaction
	path := indexPath(table, index)
	raw, _ := txn.rootTxn.Get(path)
	indexTxn := raw.(*iradix.Tree).Txn()

	// Keep this open for the duration of the txn
	txn.modified[key] = indexTxn
	return indexTxn
}

// Abort is used to cancel this transaction.
// This is a noop for read transactions.
func (txn *Txn) Abort() {
	// Noop for a read transaction
	if !txn.write {
		return
	}

	// Check if already aborted or committed
	if txn.rootTxn == nil {
		return
	}

	// Clear the txn
	txn.rootTxn = nil
	txn.modified = nil

	// Release the writer lock since this is invalid
	txn.db.writer.Unlock()
}

// Commit is used to finalize this transaction.
// This is a noop for read transactions.
func (txn *Txn) Commit() {
	// Noop for a read transaction
	if !txn.write {
		return
	}

	// Check if already aborted or committed
	if txn.rootTxn == nil {
		return
	}

	// Commit each sub-transaction scoped to (table, index)
	for key, subTxn := range txn.modified {
		path := indexPath(key.Table, key.Index)
		final := subTxn.Commit()
		txn.rootTxn.Insert(path, final)
	}

	// Update the root of the DB
	newRoot := txn.rootTxn.Commit()
	atomic.StorePointer(&txn.db.root, unsafe.Pointer(newRoot))

	// Clear the txn
	txn.rootTxn = nil
	txn.modified = nil

	// Release the writer lock since this is invalid
	txn.db.writer.Unlock()

	// Run the deferred functions, if any
	for i := len(txn.after); i > 0; i-- {
		fn := txn.after[i-1]
		fn()
	}
}

// Insert is used to add or update an object into the given table
func (txn *Txn) Insert(table string, obj interface{}) error {
	if !txn.write {
		return fmt.Errorf("cannot insert in read-only transaction")
	}

	// Get the table schema
	tableSchema, ok := txn.db.schema.Tables[table]
	if !ok {
		return fmt.Errorf("invalid table '%s'", table)
	}

	// Get the primary ID of the object
	idSchema := tableSchema.Indexes[id]
	ok, idVal, err := idSchema.Indexer.FromObject(obj)
	if err != nil {
		return fmt.Errorf("failed to build primary index: %v", err)
	}
	if !ok {
		return fmt.Errorf("object missing primary index")
	}

	// Lookup the object by ID first, to see if this is an update
	idTxn := txn.writableIndex(table, id)
	existing, update := idTxn.Get(idVal)

	// On an update, there is an existing object with the given
	// primary ID. We do the update by deleting the current object
	// and inserting the new object.
	for name, indexSchema := range tableSchema.Indexes {
		indexTxn := txn.writableIndex(table, name)

		// Determine the new index value
		ok, val, err := indexSchema.Indexer.FromObject(obj)
		if err != nil {
			return fmt.Errorf("failed to build index '%s': %v", name, err)
		}

		// Handle non-unique index by computing a unique index.
		// This is done by appending the primary key which must
		// be unique anyways.
		if ok && !indexSchema.Unique {
			val = append(val, idVal...)
		}

		// Handle the update by deleting from the index first
		if update {
			okExist, valExist, err := indexSchema.Indexer.FromObject(existing)
			if err != nil {
				return fmt.Errorf("failed to build index '%s': %v", name, err)
			}
			if okExist {
				// Handle non-unique index by computing a unique index.
				// This is done by appending the primary key which must
				// be unique anyways.
				if !indexSchema.Unique {
					valExist = append(valExist, idVal...)
				}

				// If we are writing to the same index with the same value,
				// we can avoid the delete as the insert will overwrite the
				// value anyways.
				if !bytes.Equal(valExist, val) {
					indexTxn.Delete(valExist)
				}
			}
		}

		// If there is no index value, either this is an error or an expected
		// case and we can skip updating
		if !ok {
			if indexSchema.AllowMissing {
				continue
			} else {
				return fmt.Errorf("missing value for index '%s'", name)
			}
		}

		// Update the value of the index
		indexTxn.Insert(val, obj)
	}
	return nil
}

// Delete is used to delete a single object from the given table
// This object must already exist in the table
func (txn *Txn) Delete(table string, obj interface{}) error {
	if !txn.write {
		return fmt.Errorf("cannot delete in read-only transaction")
	}

	// Get the table schema
	tableSchema, ok := txn.db.schema.Tables[table]
	if !ok {
		return fmt.Errorf("invalid table '%s'", table)
	}

	// Get the primary ID of the object
	idSchema := tableSchema.Indexes[id]
	ok, idVal, err := idSchema.Indexer.FromObject(obj)
	if err != nil {
		return fmt.Errorf("failed to build primary index: %v", err)
	}
	if !ok {
		return fmt.Errorf("object missing primary index")
	}

	// Lookup the object by ID first, check fi we should continue
	idTxn := txn.writableIndex(table, id)
	existing, ok := idTxn.Get(idVal)
	if !ok {
		return fmt.Errorf("not found")
	}

	// Remove the object from all the indexes
	for name, indexSchema := range tableSchema.Indexes {
		indexTxn := txn.writableIndex(table, name)

		// Handle the update by deleting from the index first
		ok, val, err := indexSchema.Indexer.FromObject(existing)
		if err != nil {
			return fmt.Errorf("failed to build index '%s': %v", name, err)
		}
		if ok {
			// Handle non-unique index by computing a unique index.
			// This is done by appending the primary key which must
			// be unique anyways.
			if !indexSchema.Unique {
				val = append(val, idVal...)
			}
			indexTxn.Delete(val)
		}
	}
	return nil
}

// DeleteAll is used to delete all the objects in a given table
// matching the constraints on the index
func (txn *Txn) DeleteAll(table, index string, args ...interface{}) (int, error) {
	if !txn.write {
		return 0, fmt.Errorf("cannot delete in read-only transaction")
	}

	// Get all the objects
	iter, err := txn.Get(table, index, args...)
	if err != nil {
		return 0, err
	}

	// Put them into a slice so there are no safety concerns while actually
	// performing the deletes
	var objs []interface{}
	for {
		obj := iter.Next()
		if obj == nil {
			break
		}

		objs = append(objs, obj)
	}

	// Do the deletes
	num := 0
	for _, obj := range objs {
		if err := txn.Delete(table, obj); err != nil {
			return num, err
		}
		num++
	}
	return num, nil
}

// First is used to return the first matching object for
// the given constraints on the index
func (txn *Txn) First(table, index string, args ...interface{}) (interface{}, error) {
	// Get the index value
	indexSchema, val, err := txn.getIndexValue(table, index, args...)
	if err != nil {
		return nil, err
	}

	// Get the index itself
	indexTxn := txn.readableIndex(table, indexSchema.Name)

	// Do an exact lookup
	if indexSchema.Unique && val != nil && indexSchema.Name == index {
		obj, ok := indexTxn.Get(val)
		if !ok {
			return nil, nil
		}
		return obj, nil
	}

	// Handle non-unique index by using an iterator and getting the first value
	iter := indexTxn.Root().Iterator()
	iter.SeekPrefix(val)
	_, value, _ := iter.Next()
	return value, nil
}

// LongestPrefix is used to fetch the longest prefix match for the given
// constraints on the index. Note that this will not work with the memdb
// StringFieldIndex because it adds null terminators which prevent the
// algorithm from correctly finding a match (it will get to right before the
// null and fail to find a leaf node). This should only be used where the prefix
// given is capable of matching indexed entries directly, which typically only
// applies to a custom indexer. See the unit test for an example.
func (txn *Txn) LongestPrefix(table, index string, args ...interface{}) (interface{}, error) {
	// Enforce that this only works on prefix indexes.
	if !strings.HasSuffix(index, "_prefix") {
		return nil, fmt.Errorf("must use '%s_prefix' on index", index)
	}

	// Get the index value.
	indexSchema, val, err := txn.getIndexValue(table, index, args...)
	if err != nil {
		return nil, err
	}

	// This algorithm only makes sense against a unique index, otherwise the
	// index keys will have the IDs appended to them.
	if !indexSchema.Unique {
		return nil, fmt.Errorf("index '%s' is not unique", index)
	}

	// Find the longest prefix match with the given index.
	indexTxn := txn.readableIndex(table, indexSchema.Name)
	if _, value, ok := indexTxn.Root().LongestPrefix(val); ok {
		return value, nil
	}
	return nil, nil
}

// getIndexValue is used to get the IndexSchema and the value
// used to scan the index given the parameters. This handles prefix based
// scans when the index has the "_prefix" suffix. The index must support
// prefix iteration.
func (txn *Txn) getIndexValue(table, index string, args ...interface{}) (*IndexSchema, []byte, error) {
	// Get the table schema
	tableSchema, ok := txn.db.schema.Tables[table]
	if !ok {
		return nil, nil, fmt.Errorf("invalid table '%s'", table)
	}

	// Check for a prefix scan
	prefixScan := false
	if strings.HasSuffix(index, "_prefix") {
		index = strings.TrimSuffix(index, "_prefix")
		prefixScan = true
	}

	// Get the index schema
	indexSchema, ok := tableSchema.Indexes[index]
	if !ok {
		return nil, nil, fmt.Errorf("invalid index '%s'", index)
	}

	// Hot-path for when there are no arguments
	if len(args) == 0 {
		return indexSchema, nil, nil
	}

	// Special case the prefix scanning
	if prefixScan {
		prefixIndexer, ok := indexSchema.Indexer.(PrefixIndexer)
		if !ok {
			return indexSchema, nil,
				fmt.Errorf("index '%s' does not support prefix scanning", index)
		}

		val, err := prefixIndexer.PrefixFromArgs(args...)
		if err != nil {
			return indexSchema, nil, fmt.Errorf("index error: %v", err)
		}
		return indexSchema, val, err
	}

	// Get the exact match index
	val, err := indexSchema.Indexer.FromArgs(args...)
	if err != nil {
		return indexSchema, nil, fmt.Errorf("index error: %v", err)
	}
	return indexSchema, val, err
}

// ResultIterator is used to iterate over a list of results
// from a Get query on a table.
type ResultIterator interface {
	Next() interface{}
}

// Get is used to construct a ResultIterator over all the
// rows that match the given constraints of an index.
func (txn *Txn) Get(table, index string, args ...interface{}) (ResultIterator, error) {
	// Get the index value to scan
	indexSchema, val, err := txn.getIndexValue(table, index, args...)
	if err != nil {
		return nil, err
	}

	// Get the index itself
	indexTxn := txn.readableIndex(table, indexSchema.Name)
	indexRoot := indexTxn.Root()

	// Get an interator over the index
	indexIter := indexRoot.Iterator()

	// Seek the iterator to the appropriate sub-set
	indexIter.SeekPrefix(val)

	// Create an iterator
	iter := &radixIterator{
		iter: indexIter,
	}
	return iter, nil
}

// Defer is used to push a new arbitrary function onto a stack which
// gets called when a transaction is committed and finished. Deferred
// functions are called in LIFO order, and only invoked at the end of
// write transactions.
func (txn *Txn) Defer(fn func()) {
	txn.after = append(txn.after, fn)
}

// radixIterator is used to wrap an underlying iradix iterator.
// This is much mroe efficient than a sliceIterator as we are not
// materializing the entire view.
type radixIterator struct {
	iter *iradix.Iterator
}

func (r *radixIterator) Next() interface{} {
	_, value, ok := r.iter.Next()
	if !ok {
		return nil
	}
	return value
}
