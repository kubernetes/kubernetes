package cache

import "github.com/go-git/go-git/v5/plumbing"

const (
	Byte FileSize = 1 << (iota * 10)
	KiByte
	MiByte
	GiByte
)

type FileSize int64

const DefaultMaxSize FileSize = 96 * MiByte

// Object is an interface to a object cache.
type Object interface {
	// Put puts the given object into the cache. Whether this object will
	// actually be put into the cache or not is implementation specific.
	Put(o plumbing.EncodedObject)
	// Get gets an object from the cache given its hash. The second return value
	// is true if the object was returned, and false otherwise.
	Get(k plumbing.Hash) (plumbing.EncodedObject, bool)
	// Clear clears every object from the cache.
	Clear()
}

// Buffer is an interface to a buffer cache.
type Buffer interface {
	// Put puts a buffer into the cache. If the buffer is already in the cache,
	// it will be marked as used. Otherwise, it will be inserted. Buffer might
	// be evicted to make room for the new one.
	Put(key int64, slice []byte)
	// Get returns a buffer by its key. It marks the buffer as used. If the
	// buffer is not in the cache, (nil, false) will be returned.
	Get(key int64) ([]byte, bool)
	// Clear clears every object from the cache.
	Clear()
}
