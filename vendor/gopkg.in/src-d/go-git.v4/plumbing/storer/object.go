package storer

import (
	"errors"
	"io"

	"gopkg.in/src-d/go-git.v4/plumbing"
)

var (
	//ErrStop is used to stop a ForEach function in an Iter
	ErrStop = errors.New("stop iter")
)

// EncodedObjectStorer generic storage of objects
type EncodedObjectStorer interface {
	// NewEncodedObject returns a new plumbing.EncodedObject, the real type
	// of the object can be a custom implementation or the default one,
	// plumbing.MemoryObject.
	NewEncodedObject() plumbing.EncodedObject
	// SetEncodedObject saves an object into the storage, the object should
	// be create with the NewEncodedObject, method, and file if the type is
	// not supported.
	SetEncodedObject(plumbing.EncodedObject) (plumbing.Hash, error)
	// EncodedObject gets an object by hash with the given
	// plumbing.ObjectType. Implementors should return
	// (nil, plumbing.ErrObjectNotFound) if an object doesn't exist with
	// both the given hash and object type.
	//
	// Valid plumbing.ObjectType values are CommitObject, BlobObject, TagObject,
	// TreeObject and AnyObject. If plumbing.AnyObject is given, the object must
	// be looked up regardless of its type.
	EncodedObject(plumbing.ObjectType, plumbing.Hash) (plumbing.EncodedObject, error)
	// IterObjects returns a custom EncodedObjectStorer over all the object
	// on the storage.
	//
	// Valid plumbing.ObjectType values are CommitObject, BlobObject, TagObject,
	IterEncodedObjects(plumbing.ObjectType) (EncodedObjectIter, error)
}

// DeltaObjectStorer is an EncodedObjectStorer that can return delta
// objects.
type DeltaObjectStorer interface {
	// DeltaObject is the same as EncodedObject but without resolving deltas.
	// Deltas will be returned as plumbing.DeltaObject instances.
	DeltaObject(plumbing.ObjectType, plumbing.Hash) (plumbing.EncodedObject, error)
}

// Transactioner is a optional method for ObjectStorer, it enable transaction
// base write and read operations in the storage
type Transactioner interface {
	// Begin starts a transaction.
	Begin() Transaction
}

// PackfileWriter is a optional method for ObjectStorer, it enable direct write
// of packfile to the storage
type PackfileWriter interface {
	// PackfileWriter returns a writer for writing a packfile to the storage
	//
	// If the Storer not implements PackfileWriter the objects should be written
	// using the Set method.
	PackfileWriter() (io.WriteCloser, error)
}

// EncodedObjectIter is a generic closable interface for iterating over objects.
type EncodedObjectIter interface {
	Next() (plumbing.EncodedObject, error)
	ForEach(func(plumbing.EncodedObject) error) error
	Close()
}

// Transaction is an in-progress storage transaction. A transaction must end
// with a call to Commit or Rollback.
type Transaction interface {
	SetEncodedObject(plumbing.EncodedObject) (plumbing.Hash, error)
	EncodedObject(plumbing.ObjectType, plumbing.Hash) (plumbing.EncodedObject, error)
	Commit() error
	Rollback() error
}

// EncodedObjectLookupIter implements EncodedObjectIter. It iterates over a
// series of object hashes and yields their associated objects by retrieving
// each one from object storage. The retrievals are lazy and only occur when the
// iterator moves forward with a call to Next().
//
// The EncodedObjectLookupIter must be closed with a call to Close() when it is
// no longer needed.
type EncodedObjectLookupIter struct {
	storage EncodedObjectStorer
	series  []plumbing.Hash
	t       plumbing.ObjectType
	pos     int
}

// NewEncodedObjectLookupIter returns an object iterator given an object storage
// and a slice of object hashes.
func NewEncodedObjectLookupIter(
	storage EncodedObjectStorer, t plumbing.ObjectType, series []plumbing.Hash) *EncodedObjectLookupIter {
	return &EncodedObjectLookupIter{
		storage: storage,
		series:  series,
		t:       t,
	}
}

// Next returns the next object from the iterator. If the iterator has reached
// the end it will return io.EOF as an error. If the object can't be found in
// the object storage, it will return plumbing.ErrObjectNotFound as an error.
// If the object is retreieved successfully error will be nil.
func (iter *EncodedObjectLookupIter) Next() (plumbing.EncodedObject, error) {
	if iter.pos >= len(iter.series) {
		return nil, io.EOF
	}

	hash := iter.series[iter.pos]
	obj, err := iter.storage.EncodedObject(iter.t, hash)
	if err == nil {
		iter.pos++
	}

	return obj, err
}

// ForEach call the cb function for each object contained on this iter until
// an error happends or the end of the iter is reached. If ErrStop is sent
// the iteration is stop but no error is returned. The iterator is closed.
func (iter *EncodedObjectLookupIter) ForEach(cb func(plumbing.EncodedObject) error) error {
	return ForEachIterator(iter, cb)
}

// Close releases any resources used by the iterator.
func (iter *EncodedObjectLookupIter) Close() {
	iter.pos = len(iter.series)
}

// EncodedObjectSliceIter implements EncodedObjectIter. It iterates over a
// series of objects stored in a slice and yields each one in turn when Next()
// is called.
//
// The EncodedObjectSliceIter must be closed with a call to Close() when it is
// no longer needed.
type EncodedObjectSliceIter struct {
	series []plumbing.EncodedObject
	pos    int
}

// NewEncodedObjectSliceIter returns an object iterator for the given slice of
// objects.
func NewEncodedObjectSliceIter(series []plumbing.EncodedObject) *EncodedObjectSliceIter {
	return &EncodedObjectSliceIter{
		series: series,
	}
}

// Next returns the next object from the iterator. If the iterator has reached
// the end it will return io.EOF as an error. If the object is retreieved
// successfully error will be nil.
func (iter *EncodedObjectSliceIter) Next() (plumbing.EncodedObject, error) {
	if len(iter.series) == 0 {
		return nil, io.EOF
	}

	obj := iter.series[0]
	iter.series = iter.series[1:]

	return obj, nil
}

// ForEach call the cb function for each object contained on this iter until
// an error happends or the end of the iter is reached. If ErrStop is sent
// the iteration is stop but no error is returned. The iterator is closed.
func (iter *EncodedObjectSliceIter) ForEach(cb func(plumbing.EncodedObject) error) error {
	return ForEachIterator(iter, cb)
}

// Close releases any resources used by the iterator.
func (iter *EncodedObjectSliceIter) Close() {
	iter.series = []plumbing.EncodedObject{}
}

// MultiEncodedObjectIter implements EncodedObjectIter. It iterates over several
// EncodedObjectIter,
//
// The MultiObjectIter must be closed with a call to Close() when it is no
// longer needed.
type MultiEncodedObjectIter struct {
	iters []EncodedObjectIter
	pos   int
}

// NewMultiEncodedObjectIter returns an object iterator for the given slice of
// objects.
func NewMultiEncodedObjectIter(iters []EncodedObjectIter) EncodedObjectIter {
	return &MultiEncodedObjectIter{iters: iters}
}

// Next returns the next object from the iterator, if one iterator reach io.EOF
// is removed and the next one is used.
func (iter *MultiEncodedObjectIter) Next() (plumbing.EncodedObject, error) {
	if len(iter.iters) == 0 {
		return nil, io.EOF
	}

	obj, err := iter.iters[0].Next()
	if err == io.EOF {
		iter.iters[0].Close()
		iter.iters = iter.iters[1:]
		return iter.Next()
	}

	return obj, err
}

// ForEach call the cb function for each object contained on this iter until
// an error happends or the end of the iter is reached. If ErrStop is sent
// the iteration is stop but no error is returned. The iterator is closed.
func (iter *MultiEncodedObjectIter) ForEach(cb func(plumbing.EncodedObject) error) error {
	return ForEachIterator(iter, cb)
}

// Close releases any resources used by the iterator.
func (iter *MultiEncodedObjectIter) Close() {
	for _, i := range iter.iters {
		i.Close()
	}
}

type bareIterator interface {
	Next() (plumbing.EncodedObject, error)
	Close()
}

// ForEachIterator is a helper function to build iterators without need to
// rewrite the same ForEach function each time.
func ForEachIterator(iter bareIterator, cb func(plumbing.EncodedObject) error) error {
	defer iter.Close()
	for {
		obj, err := iter.Next()
		if err != nil {
			if err == io.EOF {
				return nil
			}

			return err
		}

		if err := cb(obj); err != nil {
			if err == ErrStop {
				return nil
			}

			return err
		}
	}
}
